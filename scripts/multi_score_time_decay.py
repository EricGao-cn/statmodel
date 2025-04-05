import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.special import beta, comb
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'STHeiti', 'Heiti TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def beta_binomial_pmf(k, n, alpha, beta_param):
    """
    计算Beta-Binomial分布的概率质量函数
    P(k|n,α,β) = (n choose k) * B(k+α, n-k+β) / B(α,β)
    """
    coef = comb(n, k, exact=True)
    numerator = beta(k + alpha, n - k + beta_param)
    denominator = beta(alpha, beta_param)
    return coef * numerator / denominator

def calculate_time_decay_weights(n, alpha, beta_param, days=10):
    """
    计算时间衰减权重
    使用Beta-Binomial分布作为权重函数
    """
    weights = []
    for k in range(1, days + 1):
        weight = beta_binomial_pmf(k, n, alpha, beta_param)
        weights.append(weight)
    
    # 不进行归一化，直接返回函数值作为权重
    return np.array(weights)

def process_multi_scores_with_time_decay(file_path, n=10, alpha=2, beta_param=5):
    """
    处理多个评分指标，应用时间衰减模型
    """
    # 读取数据
    df = pd.read_excel(file_path)
    
    # 获取所有评分列名
    score_columns = [col for col in df.columns if 'score' in col.lower()]
    
    if len(score_columns) == 0:
        raise ValueError("找不到评分列")
    
    print(f"找到以下评分列: {score_columns}")
    
    # 确保有日期列
    if 'date' not in df.columns:
        # 如果没有日期列，创建一个从2025-01-29开始的日期序列
        start_date = datetime(2025, 1, 29)
        df['date'] = [start_date + timedelta(days=i) for i in range(len(df))]
    
    # 确保日期格式正确
    df['date'] = pd.to_datetime(df['date'])
    
    # 计算时间衰减权重
    weights = calculate_time_decay_weights(n, alpha, beta_param, days=10)
    print(f"时间衰减权重: {weights}")
    
    # 为每个评分列应用时间衰减
    results = {}
    
    # 只处理标准化的评分列
    standardized_columns = [col for col in score_columns if 'standardized' in col.lower()]
    if not standardized_columns:
        standardized_columns = score_columns  # 如果没有标准化列，使用所有评分列
    
    # 尝试直接处理原始数据
    # 创建一个新的DataFrame用于存储处理后的数据
    processed_df = pd.DataFrame({'date': df['date']})
    
    # 处理每个评分列
    for score_column in standardized_columns:
        # 尝试提取数值部分
        try:
            # 如果是字符串，尝试提取数值部分
            if df[score_column].dtype == 'object':
                # 使用正则表达式提取数值
                processed_df[score_column] = df[score_column].str.extract(r'([-+]?\d*\.\d+|\d+)').astype(float)
            else:
                processed_df[score_column] = df[score_column]
        except Exception as e:
            print(f"处理列 {score_column} 时出错: {e}")
            continue
    
    # 对处理后的数据应用时间衰减
    for score_column in standardized_columns:
        if score_column not in processed_df.columns:
            continue
            
        # 创建包含日期和当前评分的DataFrame
        score_df = pd.DataFrame({
            'date': processed_df['date'],
            'raw_score': processed_df[score_column]
        })
        
        # 删除NaN值
        score_df = score_df.dropna(subset=['raw_score'])
        
        if len(score_df) == 0:
            print(f"警告: {score_column}列中没有有效的数值数据，跳过此列")
            continue
        
        # 应用时间衰减
        score_df['time_decayed_score'] = np.nan
        
        for i in range(len(score_df)):
            if i < 10:  # 前10天没有足够的历史数据
                score_df.loc[i, 'time_decayed_score'] = score_df.loc[i, 'raw_score']
            else:
                # 计算时间衰减加权和
                decayed_value = 0
                for k in range(10):
                    decayed_value += weights[k] * score_df.loc[i-k-1, 'raw_score']
                score_df.loc[i, 'time_decayed_score'] = decayed_value
        
        # 存储结果
        results[score_column] = score_df
    
    # 合并所有结果
    merged_results = processed_df[['date']].copy()
    for score_column, score_df in results.items():
        # 使用merge而不是直接赋值，以处理可能的索引不匹配问题
        temp_df = pd.DataFrame({
            'date': score_df['date'],
            f"{score_column}_decayed": score_df['time_decayed_score']
        })
        merged_results = pd.merge(merged_results, temp_df, on='date', how='left')
    
    # 计算加权评分
    # 设置各评分指标的权重（保留5位小数，已归一化）
    score_weights = {
        'sentiment_score_standardized': 0.29815,
        'storytelling_score_standardized': 0.29770,
        'character_performance_score_standardized': 0.21095,
        'production_score_standardized': 0.19320
    }
    
    # 计算加权总分 - 完全重写这部分
    merged_results['weighted_score'] = 0.0
    
    # 打印所有列名以便调试
    print("合并结果中的所有列:", merged_results.columns.tolist())
    
    # 手动计算加权总分
    for idx in merged_results.index:
        weighted_sum = 0.0
        for score_column in standardized_columns:
            decayed_column = f"{score_column}_decayed"
            if decayed_column in merged_results.columns:
                value = merged_results.loc[idx, decayed_column]
                if pd.notna(value):  # 确保值不是NaN
                    weight = score_weights.get(score_column, 0)
                    weighted_sum += value * weight
        
        merged_results.loc[idx, 'weighted_score'] = weighted_sum
    
    # 保存结果
    output_file = os.path.join(os.path.dirname(file_path), "all_scores_time_decayed.xlsx")
    merged_results.to_excel(output_file, index=False)
    
    # 创建图表部分保持不变
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (score_column, score_df) in enumerate(results.items()):
        if i < 4:  # 前4个子图用于各个评分
            ax = axes[i]
            
            # 绘制原始分数和时间衰减后的分数
            ax.plot(score_df['date'], score_df['raw_score'], 'o-', color='tab:blue', label='原始分数', markersize=4)
            ax.plot(score_df['date'], score_df['time_decayed_score'], 's-', color='tab:orange', label='时间衰减后分数', markersize=4)
            
            # 设置标题和标签
            ax.set_title(f'{score_column} 时间衰减模型')
            ax.set_xlabel('日期')
            ax.set_ylabel('标准化评分')
            
            # 设置x轴日期格式
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            
            # 添加网格线和图例
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='upper right')
    
    # 第5个子图用于加权总分
    ax = axes[4]
    ax.plot(merged_results['date'], merged_results['weighted_score'], 'd-', color='tab:red', label='加权总分', markersize=4, linewidth=2)
    ax.set_title('四项评分加权总分')
    ax.set_xlabel('日期')
    ax.set_ylabel('加权评分')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right')
    
    # 隐藏第6个子图
    axes[5].axis('off')
    
    # 设置总标题
    fig.suptitle(f'四项评分指标的时间衰减模型与加权总分 (n={n}, α={alpha}, β={beta_param})', fontsize=16)
    
    # 调整布局
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图表
    chart_file = os.path.join(os.path.dirname(file_path), "all_scores_time_decay_chart.png")
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 单独绘制加权总分图表
    plt.figure(figsize=(14, 7))
    plt.plot(merged_results['date'], merged_results['weighted_score'], 'd-', color='tab:red', label='加权总分', markersize=4, linewidth=2)
    plt.title('四项评分加权总分')
    plt.xlabel('日期')
    plt.ylabel('加权评分')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    weighted_chart_file = os.path.join(os.path.dirname(file_path), "weighted_score_chart.png")
    plt.savefig(weighted_chart_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"处理完成。结果已保存至 {output_file}")
    print(f"图表已保存至 {chart_file}")
    print(f"加权总分图表已保存至 {weighted_chart_file}")
    
    return output_file, chart_file

if __name__ == "__main__":
    file_path = "data/daily_scores_summary_standardized.xlsx"
    
    try:
        # 使用指定的参数处理数据
        output_file, chart_file = process_multi_scores_with_time_decay(file_path, n=10, alpha=2, beta_param=5)
        print(f"处理成功！")
    except Exception as e:
        print(f"处理过程中出现错误: {e}")