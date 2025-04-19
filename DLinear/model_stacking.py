import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple, Union
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from darts import TimeSeries
from darts.metrics import mape, rmse, mae

# 导入基础模型
from DLinear.simple_dlinear import SimpleDLinearPredictor
from DLinear.xgboost_model import MovieBoxOfficeXGBoost

class ModelStacking:
    """
    使用模型堆叠（Model Stacking）方法结合XGBoost和DLinear模型预测票房
    
    模型堆叠是一种集成学习技术，通过训练元学习器来组合多个基础模型的预测结果
    """
    
    def __init__(
        self,
        meta_model_type: str = 'linear',  # 'linear', 'ridge', 'lasso', 'rf'
        seq_len: int = 10,
        pred_len: int = 5,
        alpha: float = 0.5,  # 用于Ridge和Lasso的正则化参数
        random_state: int = 42
    ):
        """
        初始化模型堆叠框架
        
        参数:
            meta_model_type: 元模型类型 ('linear': 线性回归, 'ridge': 岭回归, 
                            'lasso': Lasso回归, 'rf': 随机森林)
            seq_len: 输入序列长度
            pred_len: 预测序列长度
            alpha: Ridge或Lasso回归的正则化参数
            random_state: 随机种子
        """
        self.meta_model_type = meta_model_type
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.alpha = alpha
        self.random_state = random_state
        
        # 初始化基础模型
        self.xgboost_model = None
        self.dlinear_model = None
        self.meta_model = None
        
        # 初始化元模型
        if meta_model_type == 'linear':
            self.meta_model = LinearRegression()
        elif meta_model_type == 'ridge':
            self.meta_model = Ridge(alpha=alpha, random_state=random_state)
        elif meta_model_type == 'lasso':
            self.meta_model = Lasso(alpha=alpha, random_state=random_state)
        elif meta_model_type == 'rf':
            self.meta_model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        else:
            raise ValueError("不支持的元模型类型。支持的类型有: 'linear', 'ridge', 'lasso', 'rf'")
        
        # 数据缩放器
        self.scaler_target = None
    
    def initialize_base_models(
        self, 
        xgboost_params: Dict = None, 
        dlinear_params: Dict = None
    ):
        """
        初始化基础模型（XGBoost和DLinear）
        
        参数:
            xgboost_params: XGBoost模型参数
            dlinear_params: DLinear模型参数
        """
        # 使用默认参数或自定义参数初始化XGBoost模型
        if xgboost_params is None:
            xgboost_params = {
                'seq_len': self.seq_len,
                'pred_len': self.pred_len,
                'learning_rate': 0.1,
                'max_depth': 5,
                'n_estimators': 100,
                'random_state': self.random_state
            }
            
        self.xgboost_model = MovieBoxOfficeXGBoost(**xgboost_params)
        
        # 使用默认参数或自定义参数初始化DLinear模型
        if dlinear_params is None:
            dlinear_params = {
                'input_chunk_length': self.seq_len,
                'output_chunk_length': self.pred_len,
                'learning_rate': 1e-4,
                'batch_size': 32,
                'epochs': 100,
                'random_state': self.random_state
            }
            
        self.dlinear_model = SimpleDLinearPredictor(**dlinear_params)
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'box_office',
        sentiment_col: str = 'sentiment_index',
        date_col: str = 'date',
        static_cat_cols: List[str] = ['director', 'lead_actor', 'movie_type'],
        static_num_cols: List[str] = [],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Dict:
        """
        准备训练、验证和测试数据
        
        参数:
            df: 输入数据框
            target_col: 目标列名（票房）
            sentiment_col: 情感指数列名
            date_col: 日期列名
            static_cat_cols: 类别型静态特征列（导演、演员、类型等）
            static_num_cols: 数值型静态特征列
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            
        返回:
            包含训练、验证和测试数据的字典
        """
        # 确保日期列是日期时间类型
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        
        # 计算数据分割点
        n = len(df)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        # 分割数据集
        train_df = df.iloc[:train_size].copy()
        val_df = df.iloc[train_size:train_size+val_size].copy()
        test_df = df.iloc[train_size+val_size:].copy()
        
        # 为XGBoost准备数据
        target_train_xgb, sentiment_train_xgb, static_covariates_train_xgb = self.xgboost_model.prepare_data(
            train_df, 
            target_col=target_col,
            sentiment_col=sentiment_col,
            static_cat_cols=static_cat_cols,
            static_num_cols=static_num_cols
        )
        
        target_val_xgb, sentiment_val_xgb, static_covariates_val_xgb = self.xgboost_model.prepare_data(
            val_df,
            target_col=target_col,
            sentiment_col=sentiment_col,
            static_cat_cols=static_cat_cols,
            static_num_cols=static_num_cols
        )
        
        target_test_xgb, sentiment_test_xgb, static_covariates_test_xgb = self.xgboost_model.prepare_data(
            test_df,
            target_col=target_col,
            sentiment_col=sentiment_col,
            static_cat_cols=static_cat_cols,
            static_num_cols=static_num_cols
        )
        
        # 为DLinear准备数据
        target_train_dl, sentiment_train_dl = self.dlinear_model.prepare_data(
            train_df,
            target_col=target_col,
            covariate_col=sentiment_col,
            date_col=date_col
        )
        
        target_val_dl, sentiment_val_dl = self.dlinear_model.prepare_data(
            val_df,
            target_col=target_col,
            covariate_col=sentiment_col,
            date_col=date_col
        )
        
        target_test_dl, sentiment_test_dl = self.dlinear_model.prepare_data(
            test_df,
            target_col=target_col,
            covariate_col=sentiment_col,
            date_col=date_col
        )
        
        # 保存原始目标序列用于反向转换和评估
        self.scaler_target = self.xgboost_model.scaler_target
        
        # 返回准备好的数据
        return {
            'xgboost': {
                'train': (target_train_xgb, sentiment_train_xgb, static_covariates_train_xgb),
                'val': (target_val_xgb, sentiment_val_xgb, static_covariates_val_xgb),
                'test': (target_test_xgb, sentiment_test_xgb, static_covariates_test_xgb)
            },
            'dlinear': {
                'train': (target_train_dl, sentiment_train_dl),
                'val': (target_val_dl, sentiment_val_dl),
                'test': (target_test_dl, sentiment_test_dl)
            }
        }
    
    def train_base_models(self, data: Dict):
        """
        训练基础模型（XGBoost和DLinear）
        
        参数:
            data: 由prepare_data方法返回的数据字典
        """
        # 训练XGBoost模型
        xgb_train_data = data['xgboost']['train']
        self.xgboost_model.train(
            xgb_train_data[0],  # target_series
            xgb_train_data[1],  # sentiment_series
            xgb_train_data[2]   # static_covariates
        )
        print("XGBoost模型训练完成")
        
        # 训练DLinear模型
        dl_train_data = data['dlinear']['train']
        self.dlinear_model.train(
            dl_train_data[0],  # target_series
            dl_train_data[1]   # covariate_series
        )
        print("DLinear模型训练完成")
    
    def generate_meta_features(self, data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成用于训练元模型的特征
        
        参数:
            data: 由prepare_data方法返回的数据字典
            
        返回:
            X: 元特征（基础模型的预测结果）
            y: 实际目标值
        """
        # 使用验证集生成元特征
        val_data_xgb = data['xgboost']['val']
        val_data_dl = data['dlinear']['val']
        
        # 获取XGBoost在验证集上的预测
        xgb_pred = self.xgboost_model.predict(
            val_data_xgb[0],  # target_series
            val_data_xgb[1],  # sentiment_series
            val_data_xgb[2]   # static_covariates
        )
        
        # 获取DLinear在验证集上的预测
        # 为DLinear准备预测步数
        pred_length = len(val_data_dl[0]) - self.seq_len
        if pred_length <= 0:
            raise ValueError("验证集长度不足以进行预测")
            
        # 使用DLinear模型预测
        dl_pred = self.dlinear_model.predict(
            pred_length, 
            future_covariates=val_data_dl[1]
        )
        
        # 确保预测的时间戳匹配
        # 我们需要调整预测结果的时间范围，确保它们对齐
        start_idx = self.seq_len
        end_idx = len(val_data_dl[0])
        actual_values = val_data_dl[0][start_idx:end_idx]
        
        # 将预测结果和实际值转换为numpy数组
        xgb_values = xgb_pred.values().flatten()
        dl_values = dl_pred.values().flatten()
        actual_values = actual_values.values().flatten()
        
        # 确保所有数组长度一致
        min_len = min(len(xgb_values), len(dl_values), len(actual_values))
        xgb_values = xgb_values[:min_len]
        dl_values = dl_values[:min_len]
        actual_values = actual_values[:min_len]
        
        # 组合XGBoost和DLinear的预测结果作为元特征
        X_meta = np.column_stack((xgb_values, dl_values))
        y_meta = actual_values
        
        return X_meta, y_meta
    
    def train_meta_model(self, X_meta: np.ndarray, y_meta: np.ndarray):
        """
        训练元模型
        
        参数:
            X_meta: 元特征（基础模型的预测结果）
            y_meta: 实际目标值
        """
        # 训练元模型
        self.meta_model.fit(X_meta, y_meta)
        print(f"{self.meta_model_type} 元模型训练完成")
        
        # 如果是线性模型，打印权重
        if hasattr(self.meta_model, 'coef_'):
            print(f"元模型权重: XGBoost = {self.meta_model.coef_[0]:.4f}, DLinear = {self.meta_model.coef_[1]:.4f}")
            print(f"元模型截距: {self.meta_model.intercept_:.4f}")
    
    def predict(self, data: Dict) -> TimeSeries:
        """
        使用模型堆叠进行预测
        
        参数:
            data: 由prepare_data方法返回的数据字典
            
        返回:
            预测结果时间序列
        """
        # 获取测试数据
        test_data_xgb = data['xgboost']['test']
        test_data_dl = data['dlinear']['test']
        
        # 获取XGBoost在测试集上的预测
        xgb_pred = self.xgboost_model.predict(
            test_data_xgb[0],  # target_series
            test_data_xgb[1],  # sentiment_series
            test_data_xgb[2]   # static_covariates
        )
        
        # 为DLinear准备预测步数
        pred_length = len(test_data_dl[0]) - self.seq_len
        if pred_length <= 0:
            raise ValueError("测试集长度不足以进行预测")
            
        # 使用DLinear模型预测
        dl_pred = self.dlinear_model.predict(
            pred_length, 
            future_covariates=test_data_dl[1]
        )
        
        # 确保预测的时间戳匹配
        start_idx = self.seq_len
        end_idx = len(test_data_dl[0])
        
        # 获取待预测的时间索引
        time_idx = test_data_dl[0][start_idx:end_idx].time_index
        
        # 将预测结果转换为numpy数组
        xgb_values = xgb_pred.values().flatten()
        dl_values = dl_pred.values().flatten()
        
        # 确保数组长度一致
        min_len = min(len(xgb_values), len(dl_values), len(time_idx))
        xgb_values = xgb_values[:min_len]
        dl_values = dl_values[:min_len]
        time_idx = time_idx[:min_len]
        
        # 组合XGBoost和DLinear的预测结果
        X_test_meta = np.column_stack((xgb_values, dl_values))
        
        # 使用元模型进行最终预测
        stacked_pred = self.meta_model.predict(X_test_meta)
        
        # 创建新的TimeSeries对象
        stacked_pred_ts = TimeSeries.from_times_and_values(
            times=time_idx,
            values=stacked_pred.reshape(-1, 1)
        )
        
        return stacked_pred_ts
    
    def evaluate(self, actual: TimeSeries, predicted: TimeSeries) -> Dict[str, float]:
        """
        评估模型性能
        
        参数:
            actual: 实际值时间序列
            predicted: 预测值时间序列
            
        返回:
            包含评估指标的字典
        """
        # 计算评估指标
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
        xgb_pred: TimeSeries, 
        dl_pred: TimeSeries, 
        stacked_pred: TimeSeries,
        title: str = "票房预测结果对比"
    ):
        """
        绘制实际值与各模型预测值的对比图
        
        参数:
            actual: 实际值时间序列
            xgb_pred: XGBoost模型预测值
            dl_pred: DLinear模型预测值
            stacked_pred: 堆叠模型预测值
            title: 图表标题
        """
        plt.figure(figsize=(14, 7))
        actual.plot(label='实际票房')
        xgb_pred.plot(label='XGBoost预测')
        dl_pred.plot(label='DLinear预测')
        stacked_pred.plot(label='堆叠模型预测', lw=2)
        
        plt.title(title)
        plt.xlabel('日期')
        plt.ylabel('票房')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def fit(self, df: pd.DataFrame, **kwargs):
        """
        端到端训练流程，包括准备数据、训练基础模型和元模型
        
        参数:
            df: 输入数据框
            **kwargs: 传递给prepare_data方法的关键字参数
        """
        # 1. 初始化基础模型
        self.initialize_base_models()
        
        # 2. 准备数据
        data = self.prepare_data(df, **kwargs)
        
        # 3. 训练基础模型
        self.train_base_models(data)
        
        # 4. 生成元特征
        X_meta, y_meta = self.generate_meta_features(data)
        
        # 5. 训练元模型
        self.train_meta_model(X_meta, y_meta)
        
        return self


# 使用示例
def stacking_model_example():
    """
    模型堆叠示例
    """
    # 生成示例数据
    dates = pd.date_range(start='2023-01-01', periods=150)
    np.random.seed(42)
    
    # 模拟数据
    df = pd.DataFrame({
        'date': dates,
        'box_office': np.random.normal(100, 20, 150) * (1 + np.sin(np.arange(150) / 10) / 2),
        'sentiment_index': np.random.normal(0.7, 0.2, 150) + np.sin(np.arange(150) / 15) / 4,
        'director': np.random.choice(['导演A', '导演B', '导演C', '导演D'], 150),
        'lead_actor': np.random.choice(['演员A', '演员B', '演员C', '演员D', '演员E'], 150),
        'movie_type': np.random.choice(['动作', '喜剧', '科幻', '剧情', '悬疑'], 150)
    })
    
    # 创建模型堆叠实例
    stacking = ModelStacking(
        meta_model_type='ridge',  # 使用岭回归作为元模型
        seq_len=10,
        pred_len=5,
        alpha=0.01,
        random_state=42
    )
    
    # 端到端训练
    stacking.fit(df)
    
    # 准备预测数据
    data = stacking.prepare_data(df)
    
    # 获取测试数据的实际值（从序列长度之后开始）
    test_data_dl = data['dlinear']['test']
    start_idx = stacking.seq_len
    end_idx = len(test_data_dl[0])
    actual_test = test_data_dl[0][start_idx:end_idx]
    
    # 各模型单独预测
    test_data_xgb = data['xgboost']['test']
    xgb_pred = stacking.xgboost_model.predict(
        test_data_xgb[0],
        test_data_xgb[1],
        test_data_xgb[2]
    )
    
    pred_length = len(test_data_dl[0]) - stacking.seq_len
    dl_pred = stacking.dlinear_model.predict(
        pred_length,
        future_covariates=test_data_dl[1]
    )
    
    # 堆叠模型预测
    stacked_pred = stacking.predict(data)
    
    # 评估单个模型和堆叠模型的性能
    # 确保时间序列长度一致
    min_len = min(len(actual_test), len(xgb_pred), len(dl_pred), len(stacked_pred))
    actual_test_trimmed = actual_test[:min_len]
    xgb_pred_trimmed = xgb_pred[:min_len]
    dl_pred_trimmed = dl_pred[:min_len]
    stacked_pred_trimmed = stacked_pred[:min_len]
    
    # 评估性能
    xgb_metrics = stacking.evaluate(actual_test_trimmed, xgb_pred_trimmed)
    dl_metrics = stacking.evaluate(actual_test_trimmed, dl_pred_trimmed)
    stacked_metrics = stacking.evaluate(actual_test_trimmed, stacked_pred_trimmed)
    
    print("\nXGBoost模型评估:")
    for metric, value in xgb_metrics.items():
        print(f"{metric}: {value:.4f}")
        
    print("\nDLinear模型评估:")
    for metric, value in dl_metrics.items():
        print(f"{metric}: {value:.4f}")
        
    print("\n堆叠模型评估:")
    for metric, value in stacked_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 绘制结果
    stacking.plot_results(
        actual_test_trimmed,
        xgb_pred_trimmed,
        dl_pred_trimmed,
        stacked_pred_trimmed,
        title="电影票房预测模型比较"
    )


if __name__ == "__main__":
    stacking_model_example()