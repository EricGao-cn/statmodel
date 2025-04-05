import pandas as pd
from statsmodels.tsa.seasonal import STL

import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load the data from the Excel file
excel_file = pd.ExcelFile("data/daily.xlsx")
df = excel_file.parse("Sheet2")

# Convert the 'date' column to datetime objects
df['date'] = pd.to_datetime(df['date'])

# Set the 'date' column as the index
df.set_index('date', inplace=True)

# Extract the 'box' column
box = df['box']

# Perform STL decomposition
stl = STL(box, seasonal=7)  # You can adjust the seasonal parameter
res = stl.fit()

# Plot the decomposed components
plt.figure(figsize=(12, 8))

plt.subplot(411)
plt.plot(box, label='Original')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)  # 添加y=0参考线
plt.legend(loc='upper left')

plt.subplot(412)
plt.plot(res.trend, label='Trend')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)  # 添加y=0参考线
plt.legend(loc='upper left')

plt.subplot(413)
plt.plot(res.seasonal, label='Seasonal')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)  # 添加y=0参考线
plt.legend(loc='upper left')

plt.subplot(414)
plt.plot(res.resid, label='Residual')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)  # 添加y=0参考线
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()