import pandas as pd
import numpy as np

#第一列是时间，第一行是变量名（x,y)
# 使用工作区中的文件
df = pd.read_csv("QuizFiles/quiz2.csv", index_col=0, parse_dates=True)
print("原始数据:")
print(df.head())
print(f"\n数据类型:\n{df.dtypes}")

#Calculate the arithmetic returns of A and B.  Using lambda=0.94, calculate the exponentially weighted covariance matrix of A and B.  What is the exponentially weighted covariance of A,B? Round to the nearest 1e-6, in decimal format (i.e. 1.1e-5 = 0.000011)
# Calculate arithmetic returns
df_clean = df.pct_change().dropna()
print(f"\n收益率数据:\n{df_clean.head()}")
### EWMA 相关系数矩阵
import pandas as pd
import numpy as np

# ==========================================
# 计算 EWMA 协方差 
# ==========================================
X = df_clean.values
T, N = X.shape
lam = 0.94

# A. 生成权重并归一化
weights = lam ** np.arange(T - 1, -1, -1)
weights /= weights.sum()

# B. 计算加权均值并中心化
weighted_mean = (X * weights.reshape(-1, 1)).sum(axis=0)
X_centered = X - weighted_mean

# C. 得到 EWMA 协方差矩阵
cov_matrix = (X_centered.T * weights) @ X_centered
print(f"cov_matrix:\n{cov_matrix}")

# Given the exponentially weighted covariance calculated above, assuming multivariate normality and a 0 expected value, what is the VaR of A at Alpha=5% expressed as a % to the nearest 0.01%
# VaR = -z_alpha * sigma (注意：VaR是正值，表示损失)
from scipy.stats import norm

alpha = 0.05
z_alpha = norm.ppf(alpha)  # 5%分位数，约为-1.645
print(f"\nz_alpha (5%分位数): {z_alpha:.6f}")

# 正确：使用A的方差（协方差矩阵的[0,0]元素）
var_A = cov_matrix[0, 0]
sigma_A = np.sqrt(var_A)
print(f"A的方差: {var_A:.10f}")
print(f"A的标准差: {sigma_A:.6f}")

# VaR计算（假设期望为0）
VaR_A = -z_alpha * sigma_A
print(f"\nVaR of A at Alpha=5%: {VaR_A:.6f}")
print(f"VaR as percentage: {VaR_A * 100:.4f}%")
print(f"VaR rounded to 0.01%: {round(VaR_A * 100, 2)}%")

# ==========================================
# Fit a T Distribution to the returns of B
# Assume a 0 expected value (remove the mean)
# ==========================================
print("\n" + "="*70)
print("拟合T分布到B的收益率")
print("="*70)

from scipy.stats import t

# 获取B的收益率
returns_B = df_clean['B'].values
sigma_B = returns_B.std()
df_fitted, loc_fitted, scale_fitted = t.fit(returns_B, floc=0)

print(f"\n拟合的T分布参数:")
print(f"  自由度 (df): {df_fitted:.4f}")
print(f"  位置参数 (loc): {loc_fitted:.6f}")
print(f"  尺度参数 (scale): {scale_fitted:.6f}")

# 计算VaR at Alpha=5%
# VaR = -分位数(alpha)
# 获取T分布的5%分位数
t_quantile_5 = t.ppf(alpha, df_fitted)
VaR = t_quantile_5 * scale_fitted
VaR_sigma = sigma_B * np.sqrt((df_fitted - 2) / df_fitted) * t_quantile_5
print(f"VaR of B at Alpha=5%: {VaR:.6f}")
print(f"VaR of B at Alpha=5% (using sigma_B): {VaR_sigma:.6f}")