import numpy as np
from scipy.stats import beta
from scipy.stats import betabinom
from scipy.stats import poisson

import matplotlib.pyplot as plt

# 设置参数
# params = [(0.5, 0.5), (1, 1), (2, 2), (2, 5), (5, 2)]
# param_sets = [(2, 5), (1.5, 3), (1.5, 4.5), (2, 4), (1.2, 3.6), (1.2, 2.4)]
# x = np.linspace(0, 1, 100)

# # 绘制曲线
# fig, ax = plt.subplots(figsize=(8, 6))
# for a, b in param_sets:
#     pdf = beta.pdf(x, a, b)
#     ax.plot(x, pdf, label=f'a={a}, b={b}')

# # 添加标签和标题
# ax.set_xlabel('x')
# ax.set_ylabel('PDF')
# ax.set_title('Beta Distribution')
# ax.legend()

# # 显示图形
# plt.show()

n = 10  # Number of trials
param_sets = [(2, 5, n), (1.5, 3, n), (1.5, 4.5, n), (2, 4, n), (1.2, 3.6, n), (1.2, 2.4, n)]  # (alpha, beta, n)

# Generate x values (number of successes)
x = np.arange(0, n + 1)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

for a, b, n in param_sets:
    # Calculate PMF
    pmf = betabinom.pmf(x, n, a, b)
    # Plot PMF
    ax.plot(x, pmf, marker='o', linestyle='-', label=f'a={a}, b={b}, n={n}')

# Customize plot
ax.set_xlabel('Number of Successes')
ax.set_ylabel('Probability')
ax.set_title('Beta-Binomial Distribution')
ax.legend()
ax.grid(True)

# Show plot
plt.show()