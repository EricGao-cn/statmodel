import numpy as np
import pandas as pd
from pprint import pprint
from darts import TimeSeries
from darts.models import DLinearModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, rmse, mae
from darts.utils.likelihood_models import GaussianLikelihood

import matplotlib.pyplot as plt

# Import Darts libraries

class SimpleDLinearPredictor:
    """
    A simple implementation of DLinear model to predict movie box office
    using daily sentiment index as a dynamic covariate (optional).
    """
    
    def __init__(
        self,
        input_chunk_length: int = 10,
        output_chunk_length: int = 5,
        learning_rate: float = 5e-4,
        batch_size: int = 16,
        epochs: int = 100,
        random_state: int = 42
    ):
        """
        Initialize the DLinear model
        
        Parameters:
            input_chunk_length: Input sequence length
            output_chunk_length: Prediction sequence length
            learning_rate: Learning rate
            batch_size: Batch size
            epochs: Number of training epochs
            random_state: Random seed for reproducibility
        """
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        self.model = None
        self.scaler_target = None
        self.scaler_covariate = None # Will be None if covariates are not used

    def prepare_data(self, df, target_col='box', covariate_col=None, date_col='date'):
        """
        Prepare data for the DLinear model, optionally including covariates.

        Parameters:
            df: Input DataFrame
            target_col: Target column name (box office)
            covariate_col: Covariate column name (sentiment index) or None
            date_col: Date column name

        Returns:
            scaled_target: Scaled target time series
            scaled_covariate: Scaled covariate time series (or None)
            scaler_target: Fitted target scaler
            scaler_covariate: Fitted covariate scaler (or None)
        """
        # Ensure date column is datetime type
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)

        # Create target time series (box office)
        target_series = TimeSeries.from_dataframe(
            df,
            time_col=date_col,
            value_cols=target_col
        )

        # Scale target data
        scaler_target = Scaler()
        scaled_target = scaler_target.fit_transform(target_series)

        scaled_covariate = None
        scaler_covariate = None
        if covariate_col:
            # Create covariate time series (sentiment index)
            covariate_series = TimeSeries.from_dataframe(
                df,
                time_col=date_col,
                value_cols=covariate_col
            )
            # Scale covariate data
            scaler_covariate = Scaler()
            scaled_covariate = scaler_covariate.fit_transform(covariate_series)

        return scaled_target, scaled_covariate, scaler_target, scaler_covariate

    def train(self, target_series, covariate_series=None):
        """
        Train the DLinear model, optionally using covariates.

        Parameters:
            target_series: Target time series (scaled)
            covariate_series: Covariate time series (scaled) or None
        """
        # Initialize DLinear model
        self.model = DLinearModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            n_epochs=self.epochs,
            batch_size=self.batch_size,
            optimizer_kwargs={"lr": self.learning_rate},
            likelihood=GaussianLikelihood(),
            random_state=self.random_state,
            pl_trainer_kwargs={"enable_progress_bar": True} # Enable TQDM progress bar
        )

        # Train the model
        self.model.fit(
            target_series,
            future_covariates=covariate_series, # Pass None if no covariates
            verbose=False # Disable epoch-level printing
        )

    def predict(self, n_steps, future_covariates=None):
        """
        Make predictions using the trained model.

        Parameters:
            n_steps: Number of steps to predict
            future_covariates: Future covariates (scaled) for the prediction period or None

        Returns:
            Predictions in the original scale
        """
        if self.model is None:
            raise ValueError("Model needs to be trained before predicting")
        if self.scaler_target is None:
             raise ValueError("Target scaler is not fitted. Call prepare_data or set scaler_target first.")

        # Ensure future_covariates are provided if the model was trained with covariates
        # Note: Darts models often implicitly know if they were trained with covariates.
        # However, explicit checking or ensuring consistency is good practice.
        # Here, we rely on the user passing the correct future_covariates (or None).

        predictions_scaled = self.model.predict(
            n=n_steps,
            future_covariates=future_covariates, # Pass None if no covariates used/needed
            verbose=False # Disable detailed logging for historical forecasts
        )

        # Inverse transform to get predictions in original scale
        return self.scaler_target.inverse_transform(predictions_scaled)

    def evaluate(self, target_series, covariate_series=None, test_start=None):
        """
        Evaluate the model on test data using historical forecasts.

        Parameters:
            target_series: Complete target time series (original scale)
            covariate_series: Complete covariate time series (original scale) or None
            test_start: Start date/index for test data for historical forecasts

        Returns:
            Dictionary of evaluation metrics and combined predictions (original scale)
        """
        if self.model is None:
            raise ValueError("Model needs to be trained before evaluating")
        if self.scaler_target is None:
             raise ValueError("Target scaler is not fitted.")
        if covariate_series is not None and self.scaler_covariate is None:
             raise ValueError("Covariate series provided, but covariate scaler is not fitted/set.")


        # Scale the full target series
        scaled_target = self.scaler_target.transform(target_series)

        # Scale the full covariate series if provided and scaler exists
        scaled_covariate = None
        if covariate_series is not None and self.scaler_covariate is not None:
            scaled_covariate = self.scaler_covariate.transform(covariate_series)
        elif covariate_series is not None and self.scaler_covariate is None:
             print("Warning: Covariate series provided for evaluation, but no covariate scaler available. Ignoring covariates for evaluation.")
        # If covariate_series is None, scaled_covariate remains None

        # Use scaled data for historical forecasts
        # Pass scaled_covariate (which might be None) to future_covariates
        historical_preds_scaled = self.model.historical_forecasts(
            series=scaled_target,
            future_covariates=scaled_covariate, # Pass None if no covariates
            start=test_start,
            forecast_horizon=1, # Predict one step ahead
            stride=1,
            retrain=False,
            verbose=False
        )

        # Inverse transform predictions
        # historical_preds_scaled is a list of TimeSeries, inverse transform each
        predictions_original_list = [self.scaler_target.inverse_transform(pred) for pred in historical_preds_scaled]

        # Combine all predictions into one continuous TimeSeries
        if len(predictions_original_list) > 0:
            combined_predictions = predictions_original_list[0]
            for i in range(1, len(predictions_original_list)):
                # Ensure time axis continuity if possible, otherwise simple append
                try:
                    combined_predictions = combined_predictions.append(predictions_original_list[i])
                except ValueError as e:
                     print(f"Warning appending predictions, potential time gap or overlap: {e}")
                     # Fallback to simple concatenation if append fails due to time index issues
                     # This might require more robust handling depending on the exact nature of historical_forecasts output
                     combined_predictions = combined_predictions.concatenate(predictions_original_list[i], axis=0)


            # Align actual target series and predictions using slice_intersect
            # This ensures both series cover the exact same time points before metric calculation.
            target_test_intersect = target_series.slice_intersect(combined_predictions)
            predictions_intersect = combined_predictions.slice_intersect(target_test_intersect)

            # Check if intersection resulted in empty series
            if len(target_test_intersect) == 0 or len(predictions_intersect) == 0:
                 print("Warning: Target and predictions have no overlapping time period after intersection.")
                 metrics = {"MAPE": None, "RMSE": None, "MAE": None}
            else:
                # Calculate evaluation metrics using the intersected (aligned) series
                metrics = {
                    "MAPE": mape(target_test_intersect, predictions_intersect),
                    "RMSE": rmse(target_test_intersect, predictions_intersect),
                    "MAE": mae(target_test_intersect, predictions_intersect)
                }

            # Return metrics and the original combined predictions (not the intersected one)
            return metrics, combined_predictions
        else:
            # If no predictions were generated
            empty_metrics = {"MAPE": None, "RMSE": None, "MAE": None}
            # Return None for combined predictions if empty
            return empty_metrics, None


    def plot_predictions(self, train_actual, test_actual, predictions_dict, title="DLinear Movie Box Office Predictions"):
        """
        Plot actual vs predicted box office, showing training, testing, and multiple predictions.

        Parameters:
            train_actual: Actual box office time series for training period (original scale).
            test_actual: Actual box office time series for testing period (original scale).
            predictions_dict: Dictionary where keys are labels (str) and values are
                              predicted box office time series (original scale).
            title: Plot title
        """
        plt.figure(figsize=(15, 7)) # Increased figure size

        # Plot training data
        if train_actual:
            train_actual.plot(label="Training Actual")

        # Plot testing actual data
        if test_actual:
            test_actual.plot(label="Testing Actual", lw=2) # Make actual test line thicker

        # Plot predicted data from the dictionary
        if predictions_dict:
            for label, predictions in predictions_dict.items():
                if predictions is not None and len(predictions) > 0:
                     # Align prediction with test_actual for plotting using slice_intersect
                     try:
                         # Get the intersection of test_actual and predictions
                         test_actual_intersect = test_actual.slice_intersect(predictions)
                         predictions_intersect = predictions.slice_intersect(test_actual_intersect)

                         # Check if intersection is valid before plotting
                         if len(predictions_intersect) > 0:
                             predictions_intersect.plot(label=label, linestyle='--') # Use dashed lines for predictions
                         else:
                              print(f"Warning: No overlapping time period between test actuals and predictions '{label}' for plotting.")
                              # Optionally plot raw predictions if intersection fails, but might look misaligned
                              # predictions.plot(label=f"{label} (raw)", linestyle=':')

                     except Exception as e:
                          # Catch potential errors during intersection or plotting
                          print(f"Warning: Could not process or plot predictions '{label}': {e}. Plotting raw predictions if possible.")
                          try:
                              predictions.plot(label=f"{label} (raw)", linestyle=':')
                          except Exception as plot_e:
                              print(f"Failed to plot raw predictions for '{label}': {plot_e}")


                else:
                    print(f"Warning: No predictions provided for '{label}'.")
        else:
             print("Warning: No predictions dictionary provided to plot.")


        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Box Office")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5) # Add grid
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Sample code to demonstrate usage
    df = pd.read_csv("data/final_data.csv")
    # Ensure date column is datetime type and sort
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # --- Create original TimeSeries BEFORE scaling ---
    try:
        original_target_series = TimeSeries.from_dataframe(
            df, time_col='date', value_cols='box', freq='D'
        )
        original_covariate_series = TimeSeries.from_dataframe(
            df, time_col='date', value_cols='sentiment', freq='D'
        )
    except ValueError as e:
        print(f"Warning: Could not infer frequency 'D'. Trying without frequency. Error: {e}")
        original_target_series = TimeSeries.from_dataframe(df, time_col='date', value_cols='box')
        original_covariate_series = TimeSeries.from_dataframe(df, time_col='date', value_cols='sentiment')

    # --- Initialize Predictors ---
    # Common parameters
    input_chunk = 21
    output_chunk = 7
    epochs_num = 5000 # Reduced epochs for potentially faster example run

    predictor_with_cov = SimpleDLinearPredictor(
        input_chunk_length=input_chunk, output_chunk_length=output_chunk, epochs=epochs_num
    )
    predictor_no_cov = SimpleDLinearPredictor(
        input_chunk_length=input_chunk, output_chunk_length=output_chunk, epochs=epochs_num
    )

    # --- Prepare Data (Scale) ---
    # Call prepare_data once to get scaled series and scalers
    # Use predictor_with_cov instance just to call the method
    print("Preparing and scaling data...")
    scaled_target, scaled_covariate, scaler_target, scaler_covariate = predictor_with_cov.prepare_data(
        df, target_col='box', covariate_col='sentiment'
    )

    # Assign the fitted scalers to both predictor instances
    predictor_with_cov.scaler_target = scaler_target
    predictor_with_cov.scaler_covariate = scaler_covariate # Only this one needs the covariate scaler

    predictor_no_cov.scaler_target = scaler_target
    # predictor_no_cov.scaler_covariate remains None (or is not needed)

    # --- Define Split Point ---
    # Use a Timestamp that exists in the data index
    split_timestamp_str = "2025-03-04"
    try:
        # Ensure the timestamp exists or find the nearest valid point
        timestamp = pd.Timestamp(split_timestamp_str)
        if timestamp not in original_target_series.time_index:
             # Get the location of the timestamp or the next one if exact match not found
             split_point_loc = original_target_series.time_index.get_loc(timestamp, method='bfill')
             timestamp = original_target_series.time_index[split_point_loc] # Adjust to actual data timestamp
             print(f"Adjusted split timestamp to nearest available date: {timestamp.strftime('%Y-%m-%d')}")
        else:
             # split_point_loc = original_target_series.time_index.get_loc(timestamp) # No longer needed for split_after
             print(f"Using split timestamp: {timestamp.strftime('%Y-%m-%d')}")

    except KeyError:
         raise ValueError(f"Split timestamp {split_timestamp_str} or nearest equivalent not found in data index.")
    except Exception as e: # Catch other potential index errors
         raise ValueError(f"Error finding split point for timestamp {split_timestamp_str}: {e}")


    # --- Split Data using split_after ---
    # split_after(ts) returns two series: one containing data up to and including ts,
    # and one containing data after ts.
    print(f"Splitting data after timestamp: {timestamp.strftime('%Y-%m-%d')}")
    train_target_scaled, test_target_scaled = scaled_target.split_after(timestamp)
    train_target_original, test_target_original = original_target_series.split_after(timestamp)

    # Split covariates if they exist
    train_covariate_scaled, test_covariate_scaled = (None, None)
    if scaled_covariate:
        train_covariate_scaled, test_covariate_scaled = scaled_covariate.split_after(timestamp)
    
    # Original covariates split (needed for evaluation)
    train_covariate_original, test_covariate_original = (None, None)
    if original_covariate_series:
        train_covariate_original, test_covariate_original = original_covariate_series.split_after(timestamp)


    # Evaluation start point is the beginning of the test set
    if len(test_target_original) > 0:
        eval_start_timestamp = test_target_original.start_time()
        print(f"Test set starts at: {eval_start_timestamp.strftime('%Y-%m-%d')}")
    else:
        raise ValueError("Test set is empty after splitting. Adjust split point or check data.")


    # --- Check training data length ---
    min_train_len = predictor_with_cov.input_chunk_length # Both predictors have same input length
    if len(train_target_scaled) < min_train_len:
         raise ValueError(f"Training series length ({len(train_target_scaled)}) is less than input_chunk_length ({min_train_len}). Adjust split point or model parameters.")


    # --- Train Model 1 (With Covariates) ---
    print("\n--- Training Model with Covariates ---")
    if train_covariate_scaled is None:
        print("Skipping training with covariates as scaled covariates are missing.")
        metrics_cov, combined_preds_cov = {"MAPE": None, "RMSE": None, "MAE": None}, None
    else:
        # Ensure covariate series is long enough for training relative to target
        # Darts fit handles alignment, but good to be aware
        if len(train_covariate_scaled) < len(train_target_scaled):
             print(f"Warning: Scaled training covariate series (len {len(train_covariate_scaled)}) is shorter than target (len {len(train_target_scaled)}). Ensure alignment is correct.")

        try:
            predictor_with_cov.train(train_target_scaled, train_covariate_scaled)
            print("Model with covariates training finished.")

            # --- Evaluate Model 1 (With Covariates) ---
            print("\n--- Evaluating Model with Covariates ---")
            # Pass the full original series for evaluation; evaluate handles slicing/scaling internally based on test_start
            metrics_cov, combined_preds_cov = predictor_with_cov.evaluate(
                original_target_series,
                original_covariate_series, # Pass the full original covariate series
                test_start=eval_start_timestamp # Specify where historical forecasting should start
            )
            print(f"Evaluation metrics (with covariates): {metrics_cov}")
        except Exception as e:
            print(f"Error during training/evaluation with covariates: {e}")
            metrics_cov, combined_preds_cov = {"MAPE": None, "RMSE": None, "MAE": None}, None


    # --- Train Model 2 (Without Covariates) ---
    print("\n--- Training Model without Covariates ---")
    try:
        predictor_no_cov.train(train_target_scaled, None) # Pass None for covariates
        print("Model without covariates training finished.")

        # --- Evaluate Model 2 (Without Covariates) ---
        print("\n--- Evaluating Model without Covariates ---")
        # Pass the full original series for evaluation
        metrics_no_cov, combined_preds_no_cov = predictor_no_cov.evaluate(
            original_target_series,
            None, # Pass None for covariates
            test_start=eval_start_timestamp # Specify where historical forecasting should start
        )
        print(f"Evaluation metrics (no covariates): {metrics_no_cov}")
    except Exception as e:
        print(f"Error during training/evaluation without covariates: {e}")
        metrics_no_cov, combined_preds_no_cov = {"MAPE": None, "RMSE": None, "MAE": None}, None


    # --- Plot Results ---
    print("\n--- Plotting Results ---")
    predictions_to_plot = {}
    if combined_preds_cov is not None:
        predictions_to_plot["Predicted (with Covariates)"] = combined_preds_cov
    if combined_preds_no_cov is not None:
        predictions_to_plot["Predicted (no Covariates)"] = combined_preds_no_cov

    if not predictions_to_plot:
        print("No valid predictions generated from either model. Skipping plot.")
    else:
        # Use one of the predictor instances to call plot_predictions
        predictor_with_cov.plot_predictions(
            train_target_original,
            test_target_original,
            predictions_to_plot,
            title="DLinear Box Office Prediction: With vs. Without Sentiment Covariate"
        )