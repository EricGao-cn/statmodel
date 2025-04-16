import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Union, Tuple

# 导入Darts相关库
from darts import TimeSeries
from darts.models import DLinearModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, rmse, mae
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import GaussianLikelihood

class MovieBoxOfficeDLinear:
    """
    使用DLinear模型结合情感指数和电影特征预测票房
    
    DLinear是一个结合了线性层和分解的时间序列预测模型，
    本实现通过引入电影特征（导演、主演、类型）作为静态协变量来增强预测能力
    """
    
    def __init__(
        self,
        seq_len: int = 10,
        pred_len: int = 5,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        epochs: int = 100,
        random_state: int = 42
    ):
        """
        初始化DLinear模型

        参数:
            seq_len: 输入序列长度
            pred_len: 预测序列长度
            learning_rate: 学习率
            batch_size: 批次大小
            epochs: 训练轮数
            random_state: 随机种子
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        self.model = None
        self.scaler_target = None
        self.scaler_sentiment = None
        self.encoder_dict = {}  # 用于存储类别特征的编码映射
        
    def encode_categorical_features(self, data: pd.DataFrame, cat_columns: List[str]) -> pd.DataFrame:
        """
        对类别特征（如导演、主演、电影类型）进行编码
        
        参数:
            data: 包含类别特征的DataFrame
            cat_columns: 需要编码的类别特征列表
            
        返回:
            包含编码后特征的DataFrame
        """
        encoded_data = data.copy()
        
        for col in cat_columns:
            if col not in self.encoder_dict:
                # 首次编码，创建映射
                unique_values = data[col].unique()
                self.encoder_dict[col] = {val: i for i, val in enumerate(unique_values)}
            
            # 使用映射进行编码
            encoded_data[f"{col}_encoded"] = data[col].map(self.encoder_dict[col])
            
        return encoded_data
    
    def prepare_data(
        self, 
        df: pd.DataFrame,
        target_col: str = 'box_office',
        sentiment_col: str = 'sentiment_index',
        date_col: str = 'date',
        static_cat_cols: List[str] = ['director', 'lead_actor', 'movie_type'],
        static_num_cols: List[str] = []
    ) -> Tuple[TimeSeries, TimeSeries, TimeSeries]:
        """
        准备用于DLinear模型的数据
        
        参数:
            df: 输入数据框
            target_col: 目标列名（票房）
            sentiment_col: 情感指数列名
            date_col: 日期列名
            static_cat_cols: 类别型静态特征列（导演、演员、类型等）
            static_num_cols: 数值型静态特征列
            
        返回:
            target_series: 目标时间序列（票房）
            sentiment_series: 协变量时间序列（情感指数）
            static_covariates: 静态协变量时间序列
        """
        # 确保日期列是日期时间类型
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        
        # 编码类别特征
        if static_cat_cols:
            df = self.encode_categorical_features(df, static_cat_cols)
        
        # 创建目标时间序列（票房）
        target_series = TimeSeries.from_dataframe(
            df, 
            time_col=date_col, 
            value_cols=target_col
        )
        
        # 创建协变量时间序列（情感指数）
        sentiment_series = TimeSeries.from_dataframe(
            df, 
            time_col=date_col, 
            value_cols=sentiment_col
        )
        
        # 准备静态协变量
        static_cols = []
        for col in static_cat_cols:
            static_cols.append(f"{col}_encoded")
        static_cols.extend(static_num_cols)
        
        # 创建静态协变量时间序列
        if static_cols:
            static_covariates = TimeSeries.from_dataframe(
                df,
                time_col=date_col,
                value_cols=static_cols
            )
        else:
            static_covariates = None
        
        # 数据标准化
        self.scaler_target = Scaler()
        self.scaler_sentiment = Scaler()
        
        scaled_target = self.scaler_target.fit_transform(target_series)
        scaled_sentiment = self.scaler_sentiment.fit_transform(sentiment_series)
        
        return scaled_target, scaled_sentiment, static_covariates
    
    def train(
        self, 
        target_series: TimeSeries, 
        sentiment_series: TimeSeries,
        static_covariates: Optional[TimeSeries] = None
    ):
        """
        训练DLinear模型
        
        参数:
            target_series: 目标时间序列（票房）
            sentiment_series: 协变量时间序列（情感指数）
            static_covariates: 静态协变量时间序列
        """
        # 初始化DLinear模型
        self.model = DLinearModel(
            input_chunk_length=self.seq_len,
            output_chunk_length=self.pred_len,
            n_epochs=self.epochs,
            batch_size=self.batch_size,
            optimizer_kwargs={"lr": self.learning_rate},
            likelihood=GaussianLikelihood(),
            random_state=self.random_state,
            force_reset=True
        )
        
        # 训练模型
        self.model.fit(
            series=target_series,
            past_covariates=sentiment_series,
            static_covariates=static_covariates,
            verbose=True
        )
        
        print("模型训练完成")
        
    def predict(
        self, 
        target_series: TimeSeries, 
        sentiment_series: TimeSeries,
        static_covariates: Optional[TimeSeries] = None,
        n_steps: int = None
    ) -> TimeSeries:
        """
        使用训练好的模型进行预测
        
        参数:
            target_series: 目标时间序列（票房）
            sentiment_series: 协变量时间序列（情感指数）
            static_covariates: 静态协变量时间序列
            n_steps: 预测步数，默认为None（使用pred_len）
            
        返回:
            预测结果
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        # 设置预测步数
        steps = n_steps if n_steps is not None else self.pred_len
        
        # 进行预测
        prediction = self.model.predict(
            n=steps,
            series=target_series,
            past_covariates=sentiment_series,
            static_covariates=static_covariates
        )
        
        # 反向转换预测结果
        prediction = self.scaler_target.inverse_transform(prediction)
        
        return prediction
    
    def evaluate(
        self, 
        actual: TimeSeries, 
        predicted: TimeSeries
    ) -> Dict[str, float]:
        """
        评估模型性能
        
        参数:
            actual: 实际值
            predicted: 预测值
            
        返回:
            包含各种评估指标的字典
        """
        # 计算各种评估指标
        mape_score = mape(actual, predicted)
        rmse_score = rmse(actual, predicted)
        mae_score = mae(actual, predicted)
        
        metrics = {
            'MAPE': mape_score,
            'RMSE': rmse_score,
            'MAE': mae_score
        }
        
        return metrics
    
    def plot_results(
        self, 
        actual: TimeSeries, 
        predicted: TimeSeries, 
        title: str = "DLinear票房预测结果"
    ):
        """
        绘制实际值与预测值的对比图
        
        参数:
            actual: 实际值
            predicted: 预测值
            title: 图表标题
        """
        plt.figure(figsize=(12, 6))
        actual.plot(label='实际票房')
        predicted.plot(label='预测票房')
        plt.title(title)
        plt.xlabel('日期')
        plt.ylabel('票房')
        plt.legend()
        plt.grid(True)
        plt.show()


# 使用示例
def dlinear_example():
    """
    DLinear模型使用示例
    """
    # 示例数据准备 (实际使用时请替换为真实数据)
    dates = pd.date_range(start='2023-01-01', periods=100)
    np.random.seed(42)
    
    # 模拟数据
    df = pd.DataFrame({
        'date': dates,
        'box_office': np.random.normal(100, 20, 100) * (1 + np.sin(np.arange(100) / 10) / 2),
        'sentiment_index': np.random.normal(0.7, 0.2, 100) + np.sin(np.arange(100) / 15) / 4,
        'director': np.random.choice(['导演A', '导演B', '导演C', '导演D'], 100),
        'lead_actor': np.random.choice(['演员A', '演员B', '演员C', '演员D', '演员E'], 100),
        'movie_type': np.random.choice(['动作', '喜剧', '科幻', '剧情', '悬疑'], 100)
    })
    
    # 创建模型实例
    model = MovieBoxOfficeDLinear(seq_len=10, pred_len=5, epochs=50)
    
    # 准备数据
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size-10:]  # 保留一些重叠以提供序列上下文
    
    # 准备训练数据
    target_train, sentiment_train, static_covariates_train = model.prepare_data(
        train_df, 
        target_col='box_office',
        sentiment_col='sentiment_index',
        static_cat_cols=['director', 'lead_actor', 'movie_type']
    )
    
    # 训练模型
    model.train(target_train, sentiment_train, static_covariates_train)
    
    # 准备测试数据
    target_test, sentiment_test, static_covariates_test = model.prepare_data(
        test_df, 
        target_col='box_office',
        sentiment_col='sentiment_index',
        static_cat_cols=['director', 'lead_actor', 'movie_type']
    )
    
    # 预测
    prediction = model.predict(target_test, sentiment_test, static_covariates_test)
    
    # 评估模型
    actual_values = target_test[target_train.n_timesteps:]
    metrics = model.evaluate(actual_values, prediction)
    print("评估指标:", metrics)
    
    # 绘制结果
    model.plot_results(actual_values, prediction)


if __name__ == "__main__":
    dlinear_example()
