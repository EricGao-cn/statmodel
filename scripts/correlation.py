import pandas as pd
from statsmodels.tsa.seasonal import STL

import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

excel_file = pd.ExcelFile("data/daily.xlsx")
df = excel_file.parse("Sheet2").head(20)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

box = df['box']

stl = STL(box, seasonal=7)  # You can adjust the seasonal parameter
res = stl.fit()

all_scores = pd.read_excel("data/all_scores_time_decayed.xlsx").head(20)
all_scores['date'] = pd.to_datetime(all_scores['date']).dt.date
all_scores['date'] = pd.to_datetime(all_scores['date'])
all_scores.set_index('date', inplace=True)
weighted_score = all_scores['weighted_score']

# Calculate Pearson correlation
pearson_corr, _ = pearsonr(res.resid, weighted_score)
print(f"Pearson correlation: {pearson_corr}")

# Calculate Spearman correlation
spearman_corr, _ = spearmanr(res.resid, weighted_score)
print(f"Spearman correlation: {spearman_corr}")

# # Plotting the data with dual y-axes
# fig, ax1 = plt.subplots(figsize=(12, 6))

# # 设置第一个Y轴（左侧）用于残差
# color = 'tab:blue'
# ax1.set_xlabel('Date')
# ax1.set_ylabel('Residual', color=color)
# ax1.plot(res.resid.index, res.resid, label='Residual', color=color)
# ax1.tick_params(axis='y', labelcolor=color)
# ax1.grid(True, linestyle='--', alpha=0.3)

# # 创建第二个Y轴（右侧）用于加权评分
# ax2 = ax1.twinx()
# color = 'tab:red'
# ax2.set_ylabel('Weighted Score', color=color)
# ax2.plot(weighted_score.index, weighted_score, label='Weighted Score', color=color)
# ax2.tick_params(axis='y', labelcolor=color)

# # 添加标题和图例
# plt.title('Residual vs Weighted Score (Dual Y-axes)')

# # 创建包含两个线条的组合图例
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# # 调整布局
# fig.tight_layout()

# # 显示图表
# plt.savefig('correlation_dual_axes.png', dpi=300)
# plt.show()

# Perform Granger causality test
# First ensure both series have the same index
common_dates = box.index.intersection(weighted_score.index)
box_aligned = box.loc[common_dates]
weighted_score_aligned = weighted_score.loc[common_dates]

# Create a DataFrame with both series
granger_df = pd.DataFrame({
    'box': box_aligned,
    'weighted_score': weighted_score_aligned
})

# Maximum lag to test
max_lag = 3  # You can adjust this based on your data frequency

print("\nGranger Causality Tests:")
print("------------------------")

# Test if box Granger-causes weighted_score
print("Testing if box Granger-causes weighted_score:")
gc_result1 = sm.tsa.stattools.grangercausalitytests(
    granger_df[['weighted_score', 'box']], 
    maxlag=max_lag, 
    verbose=False
)
for lag in range(1, max_lag+1):
    print(f"Lag {lag}: p-value = {gc_result1[lag][0]['ssr_chi2test'][1]}")

print("\nTesting if weighted_score Granger-causes box:")
gc_result2 = sm.tsa.stattools.grangercausalitytests(
    granger_df[['box', 'weighted_score']], 
    maxlag=max_lag, 
    verbose=False
)
for lag in range(1, max_lag+1):
    print(f"Lag {lag}: p-value = {gc_result2[lag][0]['ssr_chi2test'][1]}")

# Interpreting results
print("\nInterpretation: If p-value < 0.05, we reject the null hypothesis")
print("that one variable does not Granger-cause the other.")
