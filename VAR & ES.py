import pandas as pd

### 读取数据
# 1. 设置文件路径
file_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test7_3.csv"

# 2. 读取 CSV 文件
# index_col=0 将第一列作为索引 (通常是日期列)，如果没有索引列则设置为 None
df = pd.read_csv(file_path, index_col=0)

#  最纯粹的导入
# df = pd.read_csv(file_path)
# 此时，Pandas 会非常聪明地做两件事：
# - 自动把第一行识别为列名 (Columns)，比如 'x1', 'A', 'B'。
# - 自动在最左侧生成一套 0, 1, 2, 3... 的数字索引 (RangeIndex)。

# 若此文件无表头！
#df = pd.read_csv(file_path, header=None)

# 提取为你需要的格式
# 场景 A：如果你只需要其中一列去跑拟合
#x_array = df['x1'].dropna()  # 直接变成纯净的一维 Numpy 数组

# 场景 B：如果你有多列，想整体变成矩阵去做 Copula 或协方差
#df_clean = df.dropna() # 变成 N行 x M列 的二维数组

# 3. 打印查看
print(df.head())

# 4. 计算收益率
import numpy as np 

# 算术收益率 (Arithmetic / Simple Returns) ---
# 公式: (今天价格 - 昨天价格) / 昨天价格
# 适用: 投资组合加权计算，或者向客户汇报业绩
# pct_change() 是 pandas 专门算百分比变化的函数
simple_returns = df.pct_change().dropna()

# 对数收益率 (Log / Geometric Returns) ---
# 公式: ln(今天价格 / 昨天价格) = ln(今天价格) - ln(昨天价格)
# 适用: 建模、统计分析、时间序列加总 
# shift(1) 的意思是把整列数据“向下平移一格”，也就是拿到“昨天的价格”
log_returns = np.log(df / df.shift(1)).dropna()



### 描述性分析
# --- 基础统计量 ---
# 1. 样本数量 (Count)
# 也就是有多少个交易日的数据
n_obs = log_returns.count()
print(f"样本数量: \n{n_obs}")

# 2. 均值 (Mean)
# 也就是资产的“期望收益率”
mean_val = log_returns.mean()
print(f"均值 (Mean): \n{mean_val}")

# 3. 中位数 (Median)
# 如果中位数和均值差很多，说明分布不对称
median_val = log_returns.median()
print(f"中位数 (Median): \n{median_val}")

# 4. 最小值与最大值 (Min / Max)
# 查看历史上最惨的一天和最赚的一天
min_val = log_returns.min()
max_val = log_returns.max()
print(f"最小值 (Min): \n{min_val}")
print(f"最大值 (Max): \n{max_val}")

# --- 风险统计量 ---

# 5. 标准差 (Standard Deviation / Volatility)
# 这是最核心的风险指标，代表波动率
# pandas 默认计算的是样本标准差 (N-1)
std_val = log_returns.std()
print(f"标准差 (Std): \n{std_val}")

# 6. 方差 (Variance)
# 标准差的平方，优化模型时常用
var_val = log_returns.var()
print(f"方差 (Var): \n{var_val}")

# 7. 偏度 (Skewness)
# 衡量分布是“左偏”还是“右偏”
# 负偏 (Negative): 说明经常发生暴跌 (左尾巴长)，这是风控最怕的
# 正偏 (Positive): 说明经常暴涨
skew_val = log_returns.skew()
print(f"偏度 (Skew): \n{skew_val}")

# 8. 峰度 (Kurtosis)
# 衡量有没有“肥尾” (Fat Tails)
# Pandas 算出来的是“超额峰度” (Excess Kurtosis)
# 正态分布 = 0
# 如果 > 0: 说明是尖峰肥尾，极端风险比正态分布大 (QRM 重点关注)
kurt_val = log_returns.kurtosis()
print(f"峰度 (Kurtosis): \n{kurt_val}")

# 9. 分位数 (Quantiles)
# 比如 5% 分位数，意味着有 5% 的日子收益率比这个数还低
# 这其实就是历史模拟法的 VaR (Value at Risk)
quantile_05 = log_returns.quantile(0.05) 
print(f"5% 分位数 (Historical VaR 95%): \n{quantile_05}")

# 10. 正态性检验 (Normality Test) - Jarque-Bera 测试
import scipy.stats as stats
import pandas as pd
def run_jb_test(series):
    # Jarque-Bera 测试
    # H0 (原假设): 数据服从正态分布
    # H1 (备择假设): 数据不服从正态分布
    stat, p_value = stats.jarque_bera(series)
    
    # 返回一个清晰的结论
    return pd.Series({
        'JB_Stat': stat, 
        'P_Value': p_value,
        'Is_Normal': 'Yes' if p_value > 0.05 else 'No (Fat Tails)'
    })

# --- 对每一列应用这个测试 ---
# .apply() 会自动遍历每一列
jb_results = log_returns.apply(run_jb_test)

# --- 打印结果转置一下，方便看 ---
print("正态性检验结果 (Jarque-Bera):")
print(jb_results.T)

# 如果你想程序自动告诉你下一步该干嘛
print("\n--- 风险建模建议 ---")
for col in jb_results.columns:
    if jb_results.loc['P_Value', col] < 0.05:
        print(f"[{col}]: 拒绝正态假设 -> 建议使用 T分布 (MLE) 或 历史模拟法算 VaR。")
    else:
        print(f"[{col}]: 符合正态假设 -> 可以直接使用正态分布公式算 VaR。")

# 11. AIC/BIC
import numpy as np
import pandas as pd
import scipy.stats as stats

def get_ic(ll, k, n):
    """极简计算器：传入对数似然度(ll), 参数个数(k), 样本量(n)，输出 AICc 和 BIC"""
    aic = 2 * k - 2 * ll
    aicc = aic + (2 * k * (k + 1)) / (n - k - 1) if n > k + 1 else np.nan
    bic = k * np.log(n) - 2 * ll
    return aicc, bic

def select_best_dist(series):
    x = series.dropna().values
    n = len(x)
    if n <= 4: return pd.Series(dtype=float) # 样本极度缺失的防呆保护
    
    # 1. 正态分布拟合 (k=2)
    mu, std = stats.norm.fit(x)
    ll_n = stats.norm.logpdf(x, mu, std).sum()
    aicc_n, bic_n = get_ic(ll_n, 2, n)
    
    # 2. T分布拟合 (k=3)
    df_t, loc_t, scale_t = stats.t.fit(x)
    ll_t = stats.t.logpdf(x, df_t, loc_t, scale_t).sum()
    aicc_t, bic_t = get_ic(ll_t, 3, n)
    
    # 3. 极简输出评判结果
    return pd.Series({
        'Norm_AICc': aicc_n,
        'T_AICc': aicc_t,
        'Norm_BIC': bic_n,
        'T_BIC': bic_t,
        'Winner(AICc)': 'Norm' if aicc_n < aicc_t else 'T'
    })

# 批量执行并格式化输出
# 假设你的 DataFrame 叫 log_returns
ic_results = log_returns.apply(select_best_dist).T

# 优雅的控制台输出
print("\n=== 模型选型结果 (Normal vs T-Dist) ===")
print(ic_results.round(2)) # 全局保留两位小数，保持版面干净

print("\n=== 最终风控结论 ===")
print(ic_results['Winner(AICc)'].value_counts().to_string())
        
        
        
### 拟合T分布 （即使通过JB test, 为确保计算协方差矩阵的稳定性，我们也可以拟合T分布）
import pandas as pd
import scipy.stats as stats
import numpy as np

# --- 1. 批量拟合 T分布 (MLE) ---
# 对每一列执行 fit，提取三个核心参数：自由度(Nu), 位置(Mu), 尺度(Scale)
t_params = df.apply(lambda x: pd.Series(stats.t.fit(x.dropna()), index=['Nu', 'Mu', 'Scale'])).T

# ==============================================================================
# [已注释] 去均值化 (Demean) 的正确做法 
# ⚠️ 警告：绝对不要用 floc=0 来强行锁死中心点，这会严重扭曲优化器算出的自由度！
# 如果模型或作业要求去均值，请解开下方两行注释，在物理层面减去均值后再自由拟合：
# ==============================================================================
# df_demeaned = df - df.mean()
# t_params = df_demeaned.apply(lambda x: pd.Series(stats.t.fit(x.dropna()), index=['Nu', 'Mu', 'Scale'])).T

# --- 2. 计算 T分布的理论统计量 ---
# 警告：T分布的方差不等于 Scale^2，而是被 Nu 放大了一倍； 和样本方差也不一样，因为样本方差是基于数据的，而这里我们是基于拟合的参数来计算理论方差
# 公式: Var = Scale^2 * (Nu / (Nu - 2))
# 条件: 只有 Nu > 2 时方差才有意义，否则是无穷大
t_params['Var_Theoretical'] = t_params['Scale']**2 * (t_params['Nu'] / (t_params['Nu'] - 2))
t_params['Std_Theoretical'] = np.sqrt(t_params['Var_Theoretical'])

# 公式: 超额峰度 = 6 / (Nu - 4)
# 条件: 只有 Nu > 4 时峰度才有意义
t_params['Kurt_Theoretical'] = 6 / (t_params['Nu'] - 4)

# --- 3. 结果展示 ---
print("=== T分布完整参数表 ===")
# 按照 Nu 从小到大排序，越上面的越危险(肥尾)
print(t_params.sort_values('Nu'))

# --- 4. 风险检查 (关键) ---
# 筛选出 Nu < 4 的资产 (意味着峰度爆炸，极度肥尾)
print("\n=== [警告] 极度肥尾/危险资产 (Nu < 4) ===")
risky = t_params[t_params['Nu'] < 4]

if not risky.empty:
    print(risky[['Nu', 'Kurt_Theoretical']])
    print("提示：这些资产的理论峰度可能不存在(Inf)，需改用极值理论(EVT)。")
else:
    print("无。所有资产看起来都比较正常。")
    
    

### 计算方差-协方差矩阵 (Covariance Matrix)
import pandas as pd
import numpy as np

# ==========================================
# Method 1: Drop missing values and compute sample covariance
# 1. 数据清洗 (对齐)
# ==========================================
# 只要这一行里有任何一个空值，整行删除
# 保证矩阵是正定(Positive Definite)的
df_clean = log_returns.dropna()

print(f"原始行数: {len(log_returns)}")
print(f"清洗后参与计算的行数: {len(df_clean)}")

# ==========================================
# 2. 计算样本协方差矩阵 (Pearson)
# ==========================================
# 默认 ddof=1 (分母除以 N-1)，这是无偏估计
cov_matrix = df_clean.cov()

print("\n=== 样本协方差矩阵 (Covariance) ===")
print(cov_matrix)

# ==========================================
# 3. 计算相关系数矩阵 (Correlation)
# ==========================================
# 协方差受量纲影响看不懂，相关系数(-1到1)更直观
corr_matrix = df_clean.corr()

print("\n=== 相关系数矩阵 (Correlation) ===")
print(corr_matrix)

# Method 2: Pairwise 协方差计算
# 关键区别：不要执行 dropna()!!!
# ==========================================
# Pandas 的 .cov() 只要不手动 dropna，
# 它会自动寻找每一对 (Pair) 变量的共同非空数据进行计算
pairwise_cov_matrix = log_returns.cov()
print("=== Pairwise 协方差矩阵 ===")
print(pairwise_cov_matrix)

# 计算相关系数矩阵也是一样的道理
pairwise_corr_matrix = log_returns.corr()
print("\n=== Pairwise 相关系数矩阵 ===")
print(pairwise_corr_matrix)

# 检查样本量 (Count)
# 为了让你看清楚每个格子用了多少数据，我们可以看计数矩阵
# 这是一个高级技巧，用来检查数据是否极度不平衡
n_obs = log_returns.notna().astype(int).T @ log_returns.notna().astype(int)
print("\n=== 每个协方差数据对的样本量 (N) ===")
print(n_obs)

# Method 3: EWMA 协方差矩阵 (Exponential Weighted Moving Average)
# 适用于金融时间序列，给近期数据更大权重
# EWMA 不能建立在 Pairwise 上，只能建立在 Listwise（整行删除） 或 Imputation（填补） 之上

X = df_clean.values
T, N = X.shape
lam = 0.97  # 你指定的衰减因子

# A. 生成权重 (Weights Generation)
# 生成序列: [lambda^(T-1), lambda^(T-2), ..., lambda^0]
# 结果: 最旧的数据权重最小，最新的数据权重最大(1)
weights = lam ** np.arange(T - 1, -1, -1)

# B. 权重归一化 (Normalization)
# 强制权重之和为 1，解决小样本权重泄漏问题
weights /= weights.sum()

# C. 计算加权均值 (Weighted Mean)
# 注意: 这里用的是加权后的均值，而不是简单算术平均
# reshape(-1, 1) 是为了让 (T,) 的权重能乘到 (T, N) 的矩阵上
weighted_mean = (X * weights.reshape(-1, 1)).sum(axis=0)

# D. 中心化 (Centering)
# 数据减去加权均值
X_centered = X - weighted_mean

# E. 计算加权协方差矩阵 (Weighted Covariance)
# 利用广播机制: (X_centered.T * weights) 相当于给每一列乘上权重
# 然后再和 X_centered 做矩阵乘法
cov_matrix = (X_centered.T * weights) @ X_centered

# ==========================================
# 输出结果
# ==========================================
# 转换回 DataFrame 格式方便查看
df_cov = pd.DataFrame(cov_matrix, index=df_clean.columns, columns=df_clean.columns)

print(f"=== 基于批量加权逻辑的 EWMA 协方差矩阵 (Lambda={lam}) ===")
print(f"样本量 T: {T}")
print("-" * 40)
print(df_cov)



### EWMA 相关系数矩阵
import pandas as pd
import numpy as np

# ==========================================
# 计算 EWMA 协方差 
# ==========================================
X = df_clean.values
T, N = X.shape
lam = 0.97

# A. 生成权重并归一化
weights = lam ** np.arange(T - 1, -1, -1)
weights /= weights.sum()

# B. 计算加权均值并中心化
weighted_mean = (X * weights.reshape(-1, 1)).sum(axis=0)
X_centered = X - weighted_mean

# C. 得到 EWMA 协方差矩阵
cov_matrix = (X_centered.T * weights) @ X_centered

# ==========================================
# 转换为 EWMA 相关系数矩阵 (核心步骤)
# ==========================================
# A. 提取对角线元素 (方差)
variances = np.diag(cov_matrix)

# B. 开根号得到标准差 (Volatilities)
stds = np.sqrt(variances)

# C. 计算相关系数矩阵
# 利用外积 (Outer Product) 生成分母矩阵:
# denominator[i, j] = std[i] * std[j]
denominator = np.outer(stds, stds)

# D. 逐元素相除
corr_matrix = cov_matrix / denominator

# ==========================================
# 4. 输出结果
# ==========================================
df_corr = pd.DataFrame(corr_matrix, index=df_clean.columns, columns=df_clean.columns)

print(f"=== 基于批量加权逻辑的 EWMA 相关系数矩阵 (Lambda={lam}) ===")
print(df_corr)



### 检验该协方差矩阵是否正定
import numpy as np

def check_matrix_definiteness(matrix_df, matrix_name="Matrix", tol=1e-8):
    """
    输入: Pandas DataFrame 格式的协方差或相关系数矩阵
    输出: 最小特征值，并判定其正定性
    """
    # 1. 提取底层纯数字矩阵 (剔除表头)
    # 填充可能存在的 NaN (如果是 Pairwise 极度缺失可能导致协方差算不出来)
    mat = matrix_df.fillna(0).to_numpy()
    
    # 2. 计算特征值 (极其重要：使用 eigvalsh 而不是 eigvals)
    # 因为协方差/相关系数矩阵在数学上必定是"对称矩阵 (Symmetric Matrix)"
    # eigvalsh 专门针对对称矩阵做了底层 C 语言优化，不仅速度快，而且保证算出来的特征值全是实数！
    eigenvalues = np.linalg.eigvalsh(mat)
    
    # 3. 提取最小特征值
    min_eig = np.min(eigenvalues)
    
    print(f"\n=== 检验 [{matrix_name}] 的正定性 ===")
    print(f"最小特征值 (Min Eigenvalue): {min_eig:.8f}")
    
    # 4. 判断逻辑 (引入 tol 容差来对抗浮点数误差)
    if min_eig > tol:
        print("结论: 该矩阵是 【正定矩阵 (Positive Definite, PD)】。")
        print("业务意义: 完美！矩阵满秩，可以直接投入 Cholesky 分解生成蒙特卡洛随机数。")
    elif min_eig > -tol:
        print("结论: 该矩阵是 【半正定矩阵 (Positive Semi-Definite, PSD)】。")
        print("业务意义: 处于临界状态。说明你的资产里存在极度相似的标的（比如完全线性相关的两只股票），虽然数学上没崩溃，但略有冗余。")
    else:
        print("结论: 该矩阵是 【非正定矩阵 (Non-Definite, ND)】。")
        print("业务意义: 🚨 危险！矩阵内部存在逻辑冲突。绝对不能直接用于风控模拟，必须先执行 Higham 算法 (近邻正定矩阵修复)！")
        
    return min_eig

# 1. 检验 Pairwise 相关系数矩阵 
check_matrix_definiteness(pairwise_corr_matrix, "Pairwise 相关系数矩阵 (含错位数据)")




### 若协方差矩阵不正定，使用近似正定矩阵 (Near PSD),这里用的是Rebonato & Jäckel方法
import numpy as np
import pandas as pd

def get_near_psd(covariance_matrix):
    """
    计算最近的正半定协方差矩阵 (Higham / Rebonato & Jäckel 方法的核心步骤)
    
    逻辑步骤:
    1. Cov -> Corr: 将协方差矩阵转化为相关系数矩阵。
    2. Eigendecomposition: 对相关系数矩阵进行特征分解。
    3. Truncation: 将负特征值置为 0。
    4. Reconstruction: 重组矩阵。
    5. Rescaling: 再次调整对角线，确保相关系数矩阵对角线严格为 1。
    6. Corr -> Cov: 还原回协方差矩阵。
    
    Args:
        covariance_matrix (np.array): 原始的(可能非正定的)协方差矩阵
        
    Returns:
        np.array: 修复后的正半定(PSD)协方差矩阵
    """
    # 1. 提取波动率 (Standard Deviations)
    # std = sqrt(diag(Sigma))
    std_devs = np.sqrt(np.diag(covariance_matrix))
    
    # 2. 计算相关系数矩阵 (Correlation Matrix)
    # 构造波动率倒数的对角矩阵: D^-1
    # 这一步是为了把单位归一化: Corr = D^-1 * Cov * D^-1
    inv_std_diag = np.diag(1.0 / std_devs)
    corr_matrix = inv_std_diag @ covariance_matrix @ inv_std_diag
    
    # 3. 特征值分解 (Spectral Decomposition)
    # 使用 eigh 因为矩阵是对称的，比 eig 更稳定
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    
    # 4. 修正特征值 (Fix Eigenvalues)
    # 核心逻辑: 把所有负特征值强行置为 0 (或者一个极小的 epsilon)
    eigenvalues = np.maximum(eigenvalues, 0)
    
    # 5. 重组相关系数矩阵 (Reconstruction)
    # T = Q * Lambda * Q^T
    # 此时 T 是正定的，但对角线可能不再严格等于 1
    raw_corr_psd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    # 6. 对角线再归一化 (Rescaling)
    # Rebonato & Jäckel 的关键一步: 强制把对角线拉回 1.0
    # Scaling Matrix S = diag(1 / sqrt(diag(T)))
    # Final Corr = S * T * S
    scaling_factor = 1.0 / np.sqrt(np.diag(raw_corr_psd))
    scaling_matrix = np.diag(scaling_factor)
    
    final_corr_psd = scaling_matrix @ raw_corr_psd @ scaling_matrix
    
    # 7. 还原回协方差矩阵 (Restore Covariance)
    # Cov = D * Final_Corr * D
    # 利用外积 np.outer(std, std) 可以高效还原
    final_cov_psd = np.outer(std_devs, std_devs) * final_corr_psd
    
    return final_cov_psd

psd_cov_matrix = get_near_psd(pairwise_cov_matrix)

# 转换回 DataFrame 格式方便查看
psd_cov_matrix = pd.DataFrame(psd_cov_matrix, index=df.columns, columns=df.columns)
print("=== 修复后的正半定协方差矩阵 (Near PSD Covariance Matrix) ===")
print(psd_cov_matrix)

# 验证是否成功
try:
    np.linalg.cholesky(psd_cov_matrix)
    print("\n✅ Cholesky分解成功 (矩阵已正定)")
except np.linalg.LinAlgError:
    print("\n❌ 修复失败")
    
    
### 计算修复后的相关性矩阵 
# 提取对角线并开根号得到标准差 (Volatilities)
# np.diag 获取对角线元素 (方差)
# np.sqrt 计算标准差
cov_values = psd_cov_matrix.values

# 1. 提取标准差 (Volatilities)
vols = np.sqrt(np.diag(cov_values))

# 2. 计算外积 (Outer Product) 构造分母矩阵
# 这一步生成一个矩阵 D，其中 D[i,j] = vol[i] * vol[j]
# 比双重循环快 100 倍
vol_product_matrix = np.outer(vols, vols)

# 3. 逐元素相除得到相关系数矩阵
# 注意: 这里不用矩阵乘法 @，而是点除 /
psd_corr_matrix = cov_values / vol_product_matrix

# 4. 数值清洗 (Numerical Cleanup)
# 由于浮点数精度问题，对角线可能是 1.0000000002 或 0.9999999998
# 强制设为 1.0，保证美观和逻辑严谨
np.fill_diagonal(psd_corr_matrix, 1.0)

# 5. 转换为 DataFrame 并输出
psd_corr_matrix = pd.DataFrame(psd_corr_matrix, index=df.columns, columns=df.columns)

print("=== 修复后的相关系数矩阵 (Correlation Matrix) ===")
print(psd_corr_matrix)

# ==========================================
# 6. 最终验证 (Sanity Check)
# ==========================================
# 检查是否都在 [-1, 1] 之间
is_valid_range = np.all((psd_corr_matrix >= -1.00001) & (psd_corr_matrix <= 1.00001))
print(f"\n数值范围检查 (-1 到 1): {'✅ 通过' if is_valid_range else '❌ 失败'}")



### 使用Higham方法修复相关系数矩阵，相比Rebonato & Jäckel方法，Higham方法更迭代优化，通常能得到更接近原始矩阵的结果
import numpy as np
import pandas as pd

def project_to_positive_semidefinite(matrix):
    """
    投影到正半定锥 (Project onto Positive Semi-Definite Cone)
    逻辑: 特征值分解 -> 负特征值置零 -> 重组
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # 核心: 截断负特征值
    eigenvalues = np.maximum(eigenvalues, 0)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

def project_to_unit_diagonal(matrix):
    """
    投影到单位对角线矩阵集合 (Project onto Unit Diagonal Matrices)
    逻辑: 强制将对角线元素设为 1.0
    """
    matrix_out = matrix.copy()
    np.fill_diagonal(matrix_out, 1.0)
    return matrix_out

def compute_nearest_correlation_higham(target_correlation, tol=1e-9, max_iter=1000):
    """
    Higham (2002) 算法: 计算最近的相关系数矩阵
    使用 Dykstra's Alternating Projections 方法
    """
    # 初始化差异修正矩阵 (Dykstra's correction term)
    correction_matrix = np.zeros_like(target_correlation)
    
    # 当前的最佳估计 (初始化为原始输入)
    current_correlation = target_correlation.copy()
    
    for k in range(max_iter):
        last_correlation = current_correlation.copy()
        
        # 1. 施加之前的修正量 (R_k = Y_{k-1} - \Delta S_{k-1})
        temp_matrix = last_correlation - correction_matrix
        
        # 2. 投影到 PSD 集合 (X_k = P_S(R_k))
        psd_projection = project_to_positive_semidefinite(temp_matrix)
        
        # 3. 更新修正量 (Delta S_k = X_k - R_k)
        correction_matrix = psd_projection - temp_matrix
        
        # 4. 投影到单位对角线集合 (Y_k = P_U(X_k))
        current_correlation = project_to_unit_diagonal(psd_projection)
        
        # 5. 检查收敛性 (Frobenius Norm)
        diff = np.linalg.norm(current_correlation - last_correlation, 'fro')
        if diff < tol:
            break
            
    return current_correlation

def fix_non_psd_covariance(cov_matrix_needing_fix):
    """
    主函数: 修复非正定的协方差矩阵 (通常由 Pairwise Deletion 产生)
    流程: Cov -> Vol + Corr -> Higham Fix -> New Cov
    """
    # 1. 提取标准差 (Volatility)
    # 增加微小量 epsilon 防止除以 0 (如果某资产波动率为0)
    std_devs = np.sqrt(np.diag(cov_matrix_needing_fix))
    std_devs[std_devs < 1e-8] = 1e-8
    
    # 2. 转化为相关系数矩阵
    # 利用外积构造分母矩阵 (std_i * std_j)
    vol_product_matrix = np.outer(std_devs, std_devs)
    raw_correlation = cov_matrix_needing_fix / vol_product_matrix

    # 3. 使用 Higham 算法修复相关系数矩阵 (核心步骤)
    fixed_correlation = compute_nearest_correlation_higham(raw_correlation)
    
    # 4. 还原回协方差矩阵
    # Fixed Cov = Fixed Corr * (std_i * std_j)
    fixed_covariance = fixed_correlation * vol_product_matrix
    
    return fixed_covariance

# 调用重构后的主函数
psd_cov_matrix_Higham = fix_non_psd_covariance(pairwise_cov_matrix)

# 转换回 DataFrame
psd_cov_matrix_Higham = pd.DataFrame(psd_cov_matrix_Higham, index=df.columns, columns=df.columns)

print("=== Higham 修复后的正半定协方差矩阵 ===")
print(psd_cov_matrix_Higham)



### 继续计算修复后的相关系数矩阵
# 1. 准备数据 (转为 Numpy 进行矩阵运算)
# 虽然它是 DataFrame，但为了速度和广播机制，我们取 .values
cov_values = psd_cov_matrix_Higham.values

# 2. 提取标准差 (Volatilities)
# 对角线开根号 -> 得到每个资产的波动率
vols = np.sqrt(np.diag(cov_values))

# 3. 构造分母矩阵 (Outer Product)
# 这一步生成一个矩阵 D，其中 D[i,j] = vol[i] * vol[j]
# 这是标准化协方差矩阵的关键分母
vol_product_matrix = np.outer(vols, vols)

# 4. 计算相关系数矩阵
# 逐元素相除
corr_values_higham = cov_values / vol_product_matrix

# 5. 数值清洗 (最重要的细节)
# 由于浮点数除法精度问题 (e.g., 0.0004 / (0.02*0.02) = 0.999999998)
# 必须强制将对角线设为 1.0，否则看起来很不专业
np.fill_diagonal(corr_values_higham, 1.0)

# 6. 转换回 DataFrame
# 使用原始的索引和列名
psd_corr_matrix_Higham = pd.DataFrame(
    corr_values_higham, 
    index=psd_cov_matrix_Higham.index, 
    columns=psd_cov_matrix_Higham.columns
)

print("=== Higham 修复后的相关系数矩阵 ===")
print(psd_corr_matrix_Higham)

# ==========================================
# 7. 验证 (Sanity Check)
# ==========================================
# 检查是否正定 (这是 Higham 算法保证的，但也验证一下)
try:
    np.linalg.cholesky(corr_values_higham)
    print("\n✅ 验证通过: 相关系数矩阵是正定的 (PSD)")
except np.linalg.LinAlgError:
    print("\n❌ 验证失败: 矩阵依然非正定")

# 保存结果
psd_corr_matrix_Higham.to_csv("higham_correlation_matrix.csv")



# 验证
try:
    np.linalg.cholesky(psd_cov_matrix_Higham)
    print("\n✅ 修复成功: 矩阵已正定 (Positive Definite)")
except np.linalg.LinAlgError:
    print("\n❌ 修复失败: 矩阵依然非正定")
    
    

# ==========================================
# 主成分分析 (PCA) 解释方差评估模块
# ==========================================
import numpy as np
import pandas as pd

def analyze_pca_variance(matrix, matrix_name="Matrix"):
    """
    对协方差或相关系数矩阵进行主成分分析 (PCA)，
    输出各主成分的特征值、方差解释比例及累计解释比例。
    """
    # 兼容 DataFrame 和纯 Numpy Array 输入
    mat = matrix.values if isinstance(matrix, pd.DataFrame) else matrix
        
    # 1. 计算特征值 (使用 eigvalsh 保证对称矩阵提取实数)
    eigenvalues = np.linalg.eigvalsh(mat)
    
    # 2. 倒序排列，让包含信息量最大的主成分 (PC1) 排在最前
    ev = eigenvalues[::-1]
    
    # 3. 计算每个主成分的解释方差占比 (Variance Explained)
    # 注意：如果有负特征值，这里算出来的比例也会有负数，这恰好是诊断非正定的标志
    vexp = ev / np.sum(ev)
    
    # 4. 计算累计解释方差 (Cumulative Variance Explained)
    csexp = np.round(np.cumsum(vexp), 3)
    
    # 5. 组装成 DataFrame 报表 (额外增加了 Eigenvalue 列，让底层数据一目了然)
    pca_df = pd.DataFrame({
        'PC': np.arange(1, len(ev) + 1),
        'Eigenvalue': np.round(ev, 6),
        'Explained': np.round(vexp, 6),
        'Cumulative': csexp
    })
    
    print(f"\n=== [{matrix_name}] 主成分分析 (PCA) 解释方差表 ===")
    print(pca_df.to_string(index=False))
    
    return pca_df

# ---------------------------------------------------------
# 🚀 终极实战调用：见证 Higham 算法的奇迹
# ---------------------------------------------------------

# 1. 看看修复前的原始矩阵 (一定包含可怕的负特征值)
# 假设你之前算出的带有 NA 错位的矩阵叫 pairwise_corr_matrix
# analyze_pca_variance(pairwise_corr_matrix, matrix_name="修复前: Pairwise 原始矩阵")

# 2. 看看 Higham 修复后的完美矩阵 
# 预期结果：负特征值被彻底抹平(变成0)，前几个主成分的纯度变得极高！
pca_results = analyze_pca_variance(psd_cov_matrix_Higham, matrix_name="修复后: Higham 正定矩阵")
    
    
   
### 对PSD/PD的协方差矩阵进行Cholesky分解，得到下三角矩阵 L
import numpy as np
# 假设 psd_cov_matrix_Higham 是我们修复后的协方差矩阵
try:
    L = np.linalg.cholesky(psd_cov_matrix_Higham)
    print("✅ Cholesky分解成功，得到下三角矩阵 L:")
    L = pd.DataFrame(L, index=psd_cov_matrix_Higham.index, columns=psd_cov_matrix_Higham.columns)
    print(L)
except np.linalg.LinAlgError:
    print("❌ Cholesky分解失败，矩阵可能仍然非正定")



### PCA(Principal Component Analysis) 主成分分析可以代替Cholesky分解，用于降维和风险因子提取
import numpy as np
import pandas as pd

def get_pca_simulation_matrix(cov_matrix, explained_variance_threshold=1.0):
    """
    使用 PCA (特征值分解) 生成模拟矩阵 B，代替 Cholesky 的 L。
    
    Args:
        cov_matrix (np.array): 协方差矩阵 (必须是对称的)
        explained_variance_threshold (float): 
            1.0 表示使用全部分解 (完全还原)。
            0.95 表示只保留解释了 95% 波动的主成分 (降维，去噪)。
            
    Returns:
        np.array: 矩阵 B (形状可能是 NxN 或 NxK)
        simulate formula: Y = B @ random_normal_vector
    """
    # 1. 特征值分解 (Eigenvalue Decomposition)
    # 使用 eigh (针对对称矩阵，更稳定)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 2. 排序 (Sort)
    # eigh 默认是从小到大，PCA 需要从大到小
    # [::-1] 是倒序操作
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    
    # 3. 清洗负特征值 (Fix Negative Eigenvalues)
    # 这一步让它天然支持 PSD 矩阵 (把负的变成0)
    eigenvalues = np.maximum(eigenvalues, 0)
    
    # 4. (可选) 降维截断 (Dimensionality Reduction)
    # 如果你想只保留主成分，可以在这里截断
    if explained_variance_threshold < 1.0:
        total_var = eigenvalues.sum()
        cumulative_var = np.cumsum(eigenvalues) / total_var
        
        # 找到需要保留多少个特征值
        # searchsorted 找出第一个大于 threshold 的位置
        n_components = np.searchsorted(cumulative_var, explained_variance_threshold) + 1
        
        print(f"PCA 降维: 保留前 {n_components} 个主成分 (共 {len(eigenvalues)} 个)")
        
        # 截断
        eigenvalues = eigenvalues[:n_components]
        eigenvectors = eigenvectors[:, :n_components]
        
    # 5. 生成矩阵 B
    # B = V * sqrt(Lambda)
    # 利用广播机制: eigenvectors * sqrt(eigenvalues)
    B = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    
    return B

# ==========================================
# 使用示例
# ==========================================
cov_input = psd_cov_matrix_Higham.values
B_matrix = get_pca_simulation_matrix(cov_input, explained_variance_threshold=1.0)

print("=== PCA 生成的 B 矩阵 ===")
print(pd.DataFrame(B_matrix))  # 查看全部



### 蒙特卡洛模拟法的 VaR (Monte Carlo VaR)，这里我们使用 Cholesky 分解得到 L 矩阵，生成标准正态随机数矩阵 Z，然后通过 Y = L @ Z 生成相关的随机收益率，最后根据置信水平计算 VaR 金额
import numpy as np

# ==========================================
# 0. 参数设置 (Configuration)
# ==========================================
# 假设 psd_cov_matrix_Higham 是你之前修复好的协方差矩阵 (3x3)
# 这里的 L 是下三角矩阵 (Lower Triangular Matrix)
L = np.linalg.cholesky(psd_cov_matrix_Higham)

# 资产权重: 3个资产，各占 1/3
weights = np.array([1/3, 1/3, 1/3])  # Shape: (3,)

# 组合总价值
PV = 1_000_000

# 模拟次数 (Monte Carlo Simulations)
# 10万次能保证结果在小数点后2位稳定
n_sims = 100_000

# 置信度 (95%)
alpha = 0.05

# 均值假设 (Drift Assumption)
# 这里设为 0 (3个资产的均值都是0)
# Shape: (3,) -> [0., 0., 0.]
mu = np.zeros(len(weights)) 

# ==========================================
# 1. 蒙特卡洛引擎 (MC Engine)
# ==========================================

# Step A: 生成纯净噪音 (Uncorrelated Standard Normals)
# 形状: (3, 100000) -> 3行代表资产，10万列代表模拟的天数
# 这一步对应公式中的 Z
Z = np.random.normal(0, 1, size=(len(weights), n_sims))

# Step B: 注入相关性 (Impose Correlation)
# 公式: R_sim = L @ Z
# (3,3) @ (3,100000) -> (3, 100000)
correlated_returns = L @ Z

# Step C: 加上均值 (Add Drift) -> 这里的广播机制
# mu 的原始形状是 (3,)。
# mu.reshape(-1, 1) 把它变成了 (3, 1)。
# NumPy 会自动把这 1 列复制 10万次，加到 correlated_returns 的每一列上。
# 虽然这里全是 0，不加也行，但为了代码的通用性，保留它是专业做法。
simulated_asset_returns = correlated_returns + mu.reshape(-1, 1)

# ==========================================
# 2. 计算组合盈亏 (Portfolio P&L)
# ==========================================

# Step D: 计算组合层面的收益率
# 也就是把 3 个资产的收益率加权求和
# weights (3,) @ simulated_asset_returns (3, 100000)
# 结果 sim_port_returns 形状是 (100000,) -> 代表组合在10万种情况下的收益率
sim_port_returns = weights @ simulated_asset_returns

# Step E: 转化为金额盈亏
# 这一步得到 10万个可能的盈亏金额
sim_port_pnl = PV * sim_port_returns

# ==========================================
# 3. 计算 VaR (Calculate VaR)
# ==========================================

# Step F: 排序并取分位数
# np.percentile(x, 5) 会找从小到大排在第 5% 位置的数
# 因为 VaR 通常表示为正数（亏损金额），所以我们要加负号
VaR_MC_percent = -np.percentile(sim_port_returns, alpha * 100)
VaR_MC_dollar = -np.percentile(sim_port_pnl, alpha * 100)

# ==========================================
# 4. 结果输出 (Output)
# ==========================================
print(f"--- Monte Carlo Simulation Results ---")
print(f"Simulations: {n_sims:,}")
print(f"Confidence : {1 - alpha:.0%}")
print(f"Mean Drift : 0 (Assumed)")
print("-" * 30)
print(f"MC VaR (95%) : ${VaR_MC_dollar:,.2f}")
print(f"MC VaR (%)   : {VaR_MC_percent:.4%}")



### T分布蒙特卡洛模拟法的 VaR (Monte Carlo VaR with T-distribution),这里我们使用拟合的 T分布参数生成随机数矩阵 Z，然后通过 Y = L @ Z 生成相关的随机收益率，最后根据置信水平计算 VaR 金额
import numpy as np

# ==========================================
# 0. 参数准备
# ==========================================
# 假设 nu 是你之前用 stats.t.fit 拟合出来的自由度
# 如果 nu > 30，结果基本等于正态分布；如果 nu < 5，肥尾极其显著
nu = 5.0  # 举例：设为一个较小的数来演示肥尾效果

# 其他参数保持不变
L = np.linalg.cholesky(psd_cov_matrix_Higham) 
weights = np.array([1/3, 1/3, 1/3])
PV = 1_000_000
n_sims = 100_000
alpha = 0.05
mu = np.zeros(len(weights))

# ==========================================
# 1. 蒙特卡洛引擎 (T-Distribution Version)
# ==========================================

# [修改点 1] 生成原始的 t-分布随机数 (Raw t-noise)
# 使用 nu 作为自由度参数
# 此时 Z_t_raw 的方差不是 1，而是 nu/(nu-2)
Z_t_raw = np.random.standard_t(df=nu, size=(len(weights), n_sims))

# [修改点 2] 方差调整 (Variance Adjustment)
# 这一步至关重要！我们要把 t-分布的方差强行压缩回 1
# 这样协方差矩阵 L 才能正确发挥作用
if nu > 2:
    scale_factor = np.sqrt((nu - 2) / nu)
else:
    scale_factor = 1 # 理论上 nu<=2 方差不存在，但在工程上设为1防止报错

# 得到标准化的 t-分布噪音 (方差=1，但保留了肥尾形状)
Z_t = Z_t_raw * scale_factor

# [后续步骤完全不变] 注入相关性
# R = L @ Z
correlated_returns = L @ Z_t

# 加上均值 (广播)
simulated_asset_returns = correlated_returns + mu.reshape(-1, 1)

# ==========================================
# 2. 计算组合盈亏
# ==========================================
# 加权求和
sim_port_returns = weights @ simulated_asset_returns
# 算金额
sim_port_pnl = PV * sim_port_returns

# ==========================================
# 3. 计算 VaR
# ==========================================
VaR_t_MC_percent = -np.percentile(sim_port_returns, alpha * 100)
VaR_t_MC_dollar = -np.percentile(sim_port_pnl, alpha * 100)

print(f"--- T-Student Monte Carlo Results (nu={nu}) ---")
print(f"MC VaR (95%) : ${VaR_t_MC_dollar:,.2f}")
print(f"MC VaR (%)   : {VaR_t_MC_percent:.4%}")



### 计算简单参数法下各列的VaR
import pandas as pd
import numpy as np
import scipy.stats as stats

# ==========================================
# 1. 交互式控制台：提问是否去均值
# ==========================================
print("-" * 50)
user_choice = input("⚠️ 是否需要对数据进行去均值化 (Remove Mean)? (Y/N): ").strip().upper()
demean_flag = True if user_choice == 'Y' else False
print(f"当前设置: {'[去均值化: 开启 (Mu=0)]' if demean_flag else '[去均值化: 关闭 (保留原始均值)]'}")
print("-" * 50)

alpha = 0.05

# ==========================================
# 2. 核心计算引擎 (单列处理，自带局部 dropna)
# ==========================================
def calculate_risk_metrics(series, demean=False, a=0.05):
    # 局部 dropna 提取一维数组
    data = series.dropna().to_numpy()
    
    # 获取全局固定的 Z-score (如 5% 单侧，约为 -1.6449)
    z_score = stats.norm.ppf(a)
    
# 去均值化逻辑
    if demean:
        # 1. 物理层面的去均值化
        data = data - data.mean()
        
        # 2. 正态分布：算术均值必定为 0，直接锁死 0.0
        mu_norm = 0.0
        
        # 3. T分布：🚨 绝对不要加 floc=0！
        # 让优化器自由寻找极大似然的峰值 (它会算出极其微小的非零数，完美对齐 Julia)
        nu, loc_t, scale_t = stats.t.fit(data) 
        
    else:
        mu_norm = data.mean()
        nu, loc_t, scale_t = stats.t.fit(data)
        
    std_norm = data.std(ddof=1)
    
    # --- 计算 分数与 VaR ---
    # 1. 正态 VaR
    var_normal = -(mu_norm + z_score * std_norm)
    
    # 2. T分布 VaR
    t_score = stats.t.ppf(a, df=nu)
    var_t = -(loc_t + t_score * scale_t)
    
    # 3. 比较逻辑：谁算出来的亏损金额更大 (更保守)
    larger_var = "T-Dist" if var_t > var_normal else "Normal"
    
    return pd.Series({
        'Normal_VaR': var_normal,
        'T_VaR': var_t,
        'Z_Score': z_score,
        'T_Score': t_score,
        'Larger_VaR': larger_var
    })

# ==========================================
# 3. 批量执行并展示报表
# ==========================================
# 假设你的 DataFrame 叫 log_returns
risk_report = log_returns.apply(calculate_risk_metrics, demean=demean_flag, a=alpha).T

# 格式化打印：VaR 显示百分比，Score 显示4位小数，字符串保持原样
format_dict = {
    'Normal_VaR': '{:.4%}',
    'T_VaR': '{:.4%}',
    'Z_Score': '{:.4f}',
    'T_Score': '{:.4f}',
    'Larger_VaR': '{}'
}

print("\n=== 多资产风险测算报表 (Z-Score vs T-Score 对决) ===")
print(risk_report.style.format(format_dict).to_string())



### 计算参数法的 VaR (parametric VaR / Delta Normal VaR),这里我们假设收益率服从正态分布，使用协方差矩阵计算组合波动率，再根据置信水平查 Z 分数，最后计算 VaR 金额
import numpy as np
import pandas as pd
from scipy.stats import norm

# ==========================================
# 第一步：准备输入参数 (Setup)
# ==========================================
# 1. 投资组合总市值 (Portfolio Value)
PV = 1_000_000  

# 2. 定义持有权重 (Weights) -> 对应公式里的 Gradient (∇R)
# 如果全是股票，∇R 就是资金权重。这里假设等权重持有。
# 注意：一定要用 numpy array，方便做矩阵运算
weights = np.array([1/4, 1/4, 1/4, 1/4])  # 4 个资产，每个占 25%

# 3. 设置置信水平 (Confidence Level)
alpha = 0.05  # 95% VaR

# ==========================================
# 第二步：计算核心统计量 (The "Sandwich")
# ==========================================
# 1. 计算协方差矩阵 (Sigma)
# 对应公式中的 Σ
file_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test7_3.csv"
df_clean = pd.read_csv(file_path).dropna()  # 确保没有缺失值
cov_matrix = df_clean.cov()

# 2. 计算组合方差和波动率
# 核心公式: σ² = w.T * Σ * w (三明治公式)
# 在 Python 里，@ 符号表示矩阵乘法
port_var = weights.T @ cov_matrix @ weights
port_std = np.sqrt(port_var)

# ==========================================
# 第三步：计算 VaR
# ==========================================
# 1. 查表找 Z 分数 (Inverse CDF)
# 对于 5%，norm.ppf(0.05) ≈ -1.645
z_score = norm.ppf(alpha)

# 2. 计算最终 VaR (金额)
# 公式: VaR = - PV * Z * σ
VaR_1day = - PV * z_score * port_std

print(f"---------------正态分布参数法 VaR 计算结果--------")
print(f"95% 1-Day VaR: ${VaR_1day:,.2f}")


### 计算参数法的 VaR进阶版，这里我们假设收益率服从 T分布，使用拟合的 T分布参数计算组合波动率，再根据置信水平查 T 分数，最后计算 VaR 金额
import numpy as np
import pandas as pd
from scipy import stats

# ==========================================
# 0. 数据准备与预处理 (Data Setup)
# ==========================================
# 假设 df_clean 是你的资产收益率矩阵 (行=日期, 列=资产)
# 假设 weights 是资产权重向量 (shape: N,)
# PV 是投资组合当前总价值
# alpha 是置信水平 (e.g., 0.05 for 95% VaR)

# -----------------------------------------------------------------
# [核心步骤 A] 计算协方差矩阵 (Covariance Matrix)
# -----------------------------------------------------------------
# 陷阱提示 (Pitfall Alert):
# 如果你的数据有缺失值 (NaN)，直接用 df_clean.cov() 默认是 'pairwise' (成对删除)。
# 这会导致计算出的协方差矩阵可能 "非半正定" (Non-PSD)。
# 表现为: 算出的组合方差是负数，程序报错。

# 推荐做法 1: 严格清洗 (Drop Missing) - 最安全，但丢数据
cov_matrix = df_clean.dropna().cov()

# 推荐做法 2: 指数加权 (EWMA) - 业界标准 (RiskMetrics)，且天然PSD
# cov_matrix = df_clean.ewm(span=30).cov().iloc[-1] 

# 不推荐做法: 直接 df.cov() (Pairwise) 除非你后续做了 Higham Fix 修正

# -----------------------------------------------------------------
# [核心步骤 B] 计算组合波动率 (Portfolio Volatility - Sigma_p)
# -----------------------------------------------------------------
# 使用 "三明治公式" (Quadratic Form): w.T * Sigma * w
port_var = weights.T @ cov_matrix @ weights

# 检查非正定性 (Sanity Check for PSD)
if port_var < 0:
    raise ValueError("Error: 协方差矩阵非正定，导致方差为负。请检查数据清洗方式(比如用了Pairwise?)或使用 Higham Fix。")

port_std = np.sqrt(port_var)
print(f"\n组合正态波动率 (Sigma_p): {port_std:.4%}")


# ==========================================
# 1. 拟合 t-分布 (Fit t-distribution)
# ==========================================

# -----------------------------------------------------------------
# [关键动作] 先捏合成一个整体 (Aggregate First)
# -----------------------------------------------------------------
# 为什么？因为我们很难直接拟合多元 t-分布 (Multivariate t)。
# 我们利用线性性质，先算出 "如果持有这个权重，历史上的每天表现如何"。
# 这样就把多维问题降维成了 "一维 (Univariate)" 问题。
port_history_returns = df_clean @ weights 

# 拟合参数
# scipy.stats.t.fit 使用极大似然估计 (MLE)
# 返回: df (自由度), loc (均值), scale (标准差)
# 我们最关心 df (nu)，它决定了尾巴有多肥。
nu, loc, scale = stats.t.fit(port_history_returns)

print(f"\n拟合出的 t-分布自由度 (nu): {nu:.2f}")
if nu < 5:
    print("  -> 警告: 尾部极肥 (Fat Tails)，正态分布将严重低估风险！")
elif nu > 30:
    print("  -> 提示: 接近正态分布。")


# ==========================================
# 2. 计算修正后的乘数 (The Adjusted Multiplier)
# ==========================================

# 第一步：查 t-分布表 (Raw t-score)
# 查找 t 分布下的 alpha 分位数 (比如 5%)
t_score = stats.t.ppf(alpha, nu)

# 第二步：方差调整 (Variance Adjustment) --- 容易被遗漏！
# 逻辑：标准 t-分布的方差是 nu/(nu-2)，总是大于 1 的。
# 我们之前算的 port_std 是基于协方差矩阵的 "真实波动率"。
# 为了不重复计算波动率，我们需要把 t-分布 "缩放" 回单位方差。
if nu > 2:
    adj_factor = np.sqrt((nu - 2) / nu)
else:
    adj_factor = 1 # 极端情况 (nu<=2 意味着方差无穷大，理论崩塌)

# 最终乘数
final_multiplier = t_score * adj_factor

# ==========================================
# 3. 计算 t-VaR (Final Calculation)
# ==========================================
VaR_t_dist_percentage = - final_multiplier * port_std  # 这是一个正数，表示损失的百分比
# 公式: VaR = - PV * (调整后的乘数) * 组合波动率
VaR_t_dist = - PV * final_multiplier * port_std

print(f"\n================ t分布的参数法VaR结果 ================")
print(f"one-day t-Student VaR in percentage (95%): {VaR_t_dist_percentage:.2%}")
print(f"one-day t-Student VaR (95%): ${VaR_t_dist:,.2f}")



### 历史模拟法的 VaR (Historical Simulation VaR)，这里我们直接用历史数据计算组合的每日盈亏，然后根据置信水平取分位数计算 VaR 金额
import numpy as np

# ==========================================
# 0. 准备工作
# ==========================================
# df_clean: 你的历史收益率数据 (N行 x M列)
# weights:  你今天的持仓权重 (M列)
# PV:       本金 (比如 1,000,000)
# alpha:    0.05 (95% 置信度)

# ==========================================
# 1. 构造组合的历史收益率 (Construct Portfolio History)
# ==========================================
# 利用矩阵乘法，瞬间算出每一天的组合收益率
# 这一步就是 "Re-valuation"
print(df_clean.head())  # 查看前几行，确认数据格式正确
weights = [1/4, 1/4, 1/4, 1/4]  # 确保权重是 numpy array 或 list，长度与 df_clean 列数一致
port_hist_returns = df_clean @ weights

# ==========================================
# 2. 排序与找位 (Sort and Cut)
# ==========================================
# 方法 A: 手动排序法 (最符合 PPT 原理)
sorted_returns = np.sort(port_hist_returns) # 从小到大排
index_cutoff = int(len(sorted_returns) * alpha) # 算出第 5% 是第几个 (比如第25个)
VaR_return = sorted_returns[index_cutoff] # 取出那个数

# 方法 B: 自动分位法 (工程常用，结果更精确，自带插值)
# percentle(5) 自动帮你完成排序和找位置
VaR_return_auto = np.percentile(port_hist_returns, alpha * 100) 

# ==========================================
# 3. 算出最终金额
# ==========================================
# VaR 通常表示为正数 (Loss Amount)
VaR_dollar = -VaR_return_auto * PV 

print(f"/n历史模拟法 VaR (95%): ${VaR_dollar:,.2f}")
print(f"对应收益率: {VaR_return_auto:.4%}")



### 加权历史模拟法的 VaR (Weighted Historical Simulation VaR)，这里我们给历史数据中的每一天赋予一个权重，通常是指数衰减权重，然后根据加权的历史盈亏计算 VaR 金额
import numpy as np
import pandas as pd

# ==========================================
# 0. 参数准备
# ==========================================
# lambda_param: 衰减因子 (0.98 是业界标准)
lambda_param = 0.98
alpha = 0.05 # 95% VaR
weights = np.array([1/4, 1/4, 1/4, 1/4]) # 资产持仓权重

# 1. 穿越回过去 (Re-valuation) - 和基础版一样
# 得到每一天的组合收益率
historical_returns = df_clean @ weights

# ==========================================
# 2. 计算时间权重 (Time Weights) - 复杂版核心
# ==========================================
n_days = len(historical_returns)

# 生成一个从 0 到 n-1 的序列
# 假设 index 0 是最久远的数据，index -1 是昨天
# 我们需要反过来，让昨天 (newest) 的权重最大
time_decay = np.array([lambda_param**(n_days - 1 - i) for i in range(n_days)])

# 归一化：保证权重之和为 1
# time_weights 里的数越往后(越新)越大
time_weights = time_decay / np.sum(time_decay)

# ==========================================
# 3. 排序并绑定权重 (Sort Returns & Weights)
# ==========================================
# 这里用 pandas DataFrame 处理会最方便，因为排序时权重必须跟着收益率一起动
df_sim = pd.DataFrame({
    'returns': historical_returns,
    'weights': time_weights
})

# 按收益率从小到大排序 (最惨的亏损排在最前面)
df_sorted = df_sim.sort_values(by='returns', ascending=True)

# ==========================================
# 4. 累加权重找 5% (Cumulative Sum)
# ==========================================
# 计算累积权重
df_sorted['cum_weights'] = df_sorted['weights'].cumsum()

# 找到第一个累积权重超过 5% (alpha) 的位置
# 这个位置对应的收益率就是 VaR
var_row = df_sorted[df_sorted['cum_weights'] >= alpha].iloc[0]

VaR_weighted_return = var_row['returns']
VaR_weighted_dollar = -VaR_weighted_return * PV

print(f"--- Weighted Historical Simulation (Lambda={lambda_param}) ---")
print(f"Weighted VaR (95%): ${VaR_weighted_dollar:,.2f}")
print(f"Weighted VaR (%):   {-VaR_weighted_return:.4%}")



### Hull-White (Volatility-Weighted Historical Simulation)，这里我们根据每一天的历史波动率来调整那一天的权重，波动率越大，权重越大，然后根据加权的历史盈亏计算 VaR 金额
import numpy as np
import pandas as pd

# ==========================================
# 0. 准备数据
# ==========================================
# 假设 port_history_returns 是一个 Pandas Series
# 包含了过去 500 天组合的每日收益率
# port_history_returns = df_clean @ weights 

# 参数设置
lambda_param = 0.94  # RiskMetrics 标准衰减因子
alpha = 0.05         # 95% VaR

# ==========================================
# 1. 计算历史波动率序列 (Sigma_t)
# ==========================================
# 我们使用 EWMA (指数加权移动平均) 来估计每一天的波动率
# pandas 的 ewm().std() 可以直接算出来
# adjust=False 是为了匹配 RiskMetrics 的递归逻辑
vol_series = port_history_returns.ewm(alpha=(1 - lambda_param), adjust=False).std()

# 获取 "今天" (最新一天) 的波动率
current_vol = vol_series.iloc[-1]

print(f"当前波动率 (Current Vol): {current_vol:.4%}")

# ==========================================
# 2. 核心步骤：缩放历史收益率 (Scaling)
# ==========================================
# 公式: R_adj = R_t * (Current_Vol / Vol_t)
# 这一步把过去所有的收益率都 "标准化" 到今天的波动率水平了
scaling_factors = current_vol / vol_series
adjusted_returns = port_history_returns * scaling_factors

# [数据清洗] 去除前几天因为 EWMA 预热可能产生的 NaN
adjusted_returns = adjusted_returns.dropna()

# ==========================================
# 3. 排序并找 VaR (Sort & Cut)
# ==========================================
# 对 "调整后" 的收益率进行排序
VaR_HW_percent = -np.percentile(adjusted_returns, alpha * 100)

# 计算金额
VaR_HW_dollar = VaR_HW_percent * PV

print(f"--- Hull-White (Vol-Weighted) Results ---")
print(f"HW VaR (95%) : ${VaR_HW_dollar:,.2f}")
print(f"HW VaR (%)   : {VaR_HW_percent:.4%}")



### 构建T回归，用MLE方法估计回归参数
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize

# ==========================================
# 0. 准备数据 (假设 df 包含 y, x2, x3)
# ==========================================
# 确保你的数据中没有 NaN，如果有，需要先处理掉 (比如 dropna 或者填充)
Y = df['y'].values
# 加上截距项 (Constant), 变成 [1, x2, x3]
X = sm.add_constant(df[['x2', 'x3']]).values 

# 动态获取 Beta 的数量 (这里是 3: Intercept, B_x2, B_x3)
num_betas = X.shape[1]

# ==========================================
# 1. 定义负对数似然函数 (NLL)
# ==========================================
def nll(params):
    # --- 参数拆解 ---
    # 前 n 个参数是回归系数 (Betas)
    betas = params[:num_betas]
    # 倒数第 2 个是自由度 (Nu)
    nu = params[-2]
    # 最后一个是尺度 (Sigma)
    sigma = params[-1]
    
    # --- 核心约束 (Critical for Risk) ---
    # 1. Sigma 必须 > 0
    # 2. Nu 必须 > 2 (建议 2.1)，否则 T分布的方差无穷大，风控模型会崩
    if nu <= 2.01 or sigma <= 0:
        return np.inf

    # --- 计算残差 ---
    residuals = Y - (X @ betas)
    
    # --- 计算似然 (Log-Likelihood) ---
    # 使用 t.logpdf 计算在当前 parameters 下，残差出现的概率密度对数和
    # 取负号，因为 minimize 寻找最小值，等同于 maximize likelihood
    ll = np.sum(stats.t.logpdf(residuals, df=nu, loc=0, scale=sigma))
    return -ll

# ==========================================
# 2. 聪明的初始化 (Smart Initialization)
# ==========================================

# Step A: 先跑 OLS (假设正态) 得到 Beta 的近似解
# 这一步是为了防止 MLE 在错误的起点迷路
ols_model = sm.OLS(Y, X).fit()
beta_init = ols_model.params

# Step B: 用 OLS 的残差去拟合 T分布，得到 Nu 和 Sigma 的近似解
nu_init, _, sigma_init = stats.t.fit(ols_model.resid)

# Step C: 拼装初始参数 [Beta0, Beta1, Beta2, Nu, Sigma]
initial_params = np.append(beta_init, [nu_init, sigma_init])

# ==========================================
# 3. 执行 MLE 优化
# ==========================================
# 设定边界 (Bounds):
# Betas: 无限制 (None, None)
# Nu:    必须 > 2.1 (这是与学术界做纯研究唯一的区别，风控必须保方差)
# Sigma: 必须 > 0.0001 (防止除零错误)
bounds = [(None, None)] * num_betas + [(2.1, None), (0.0001, None)]

# 调用求解器
result = minimize(nll, initial_params, method='L-BFGS-B', bounds=bounds)

# ==========================================
# 4. 提取与展示结果
# ==========================================
# 提取最终参数
final_betas = result.x[:num_betas]
final_nu = result.x[-2]
final_sigma = result.x[-1]

print("="*40)
print("t-Regression (MLE) 结果报告")
print("="*40)
print(f"【回归系数 Betas】")
# 对应 X 的列名
param_names = ['Intercept'] + list(df[['x2', 'x3']].columns)
for name, val in zip(param_names, final_betas):
    print(f"{name:<12}: {val:.6f}")

print("-" * 40)
print(f"【残差分布参数】")
print(f"自由度 (Nu)   : {final_nu:.4f}")
if final_nu < 4:
    print("  -> 警告: 极度肥尾 (Nu < 4)，峰度理论值可能不存在。")
elif final_nu > 30:
    print("  -> 提示: 接近正态分布。")

print(f"尺度 (Sigma)  : {final_sigma:.6f}")

# (可选) 计算残差的标准差（剔除X影响后的特质波动）
if final_nu > 2:
    theo_var = final_sigma**2 * (final_nu / (final_nu - 2))
    print(f"残差的理论波动率(Std): {np.sqrt(theo_var):.6f}")
print("="*40)



### OLS回归 (假设正态分布)
import pandas as pd
import statsmodels.api as sm
import numpy as np

# ==========================================
# 0. 准备数据 & 清洗
# ==========================================
# 必须先处理空值，否则 OLS 会报错或结果不对
data = log_returns[['y', 'x2', 'x3']].dropna()

Y = data['y']
X = data[['x2', 'x3']]

# ==========================================
# 1. 核心步骤: 添加截距项 (Constant)
# ==========================================
# 这一步非常重要！如果不加，回归线会被强制经过原点 (0,0)，导致巨大的偏差。
# add_constant 会在 X 里加一列全为 1 的数据
X_with_const = sm.add_constant(X)

# ==========================================
# 2. 拟合模型 (Fit OLS)
# ==========================================
# 注意顺序: sm.OLS(Y, X) -> Y 在前，X 在后
model = sm.OLS(Y, X_with_const).fit()

# ==========================================
# 3. 打印结果 (Summary)
# ==========================================
print(model.summary())

# ==========================================
# 4. 提取关键指标 (给下游风控模型用)
# 各测验说明（都是针对残差的）：
# JB_Stat: Jarque-Bera 统计量， <0.05 拒绝正态假设
# Durbin-Watson: 1.5-2.5 之间说明残差无自相关
# ==========================================
print("\n=== 关键指标提取 ===")
print(f"Alpha (截距):     {model.params['const']:.6f}")
print(f"Beta (x2):       {model.params['x2']:.6f}")
print(f"Beta (x3):       {model.params['x3']:.6f}")
print("-" * 30)
print(f"R-squared (R方): {model.rsquared:.6f} (因子解释了多少波动)")

# 提取残差的标准差 (这就是 OLS 下的 'Sigma')
# 也就是你之前问的 'theo_var' 的正态版本
residual_std = np.std(model.resid)
print(f"残差波动率 (Std): {residual_std:.6f} (假设正态分布)")



### 计算Expected Shortfall (ES)，这里我们先假设收益率服从正态分布，根据 VaR 的 Z 分数和组合波动率计算 ES 金额
import pandas as pd
import numpy as np
from scipy.stats import norm

# 1. 读取并清理数据 (去除缺失值)
df = pd.read_csv(r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test7_1.csv")
x = df['x1'].dropna().values

# 2. 提取分布参数 (ddof=1 确保计算的是样本标准差)
mu = np.mean(x)
sigma = np.std(x, ddof=1)
alpha = 0.05

# 3. 正态分布计算核心
z = norm.ppf(alpha)      # 5% 尾部的临界 Z-score (负数)
pdf_z = norm.pdf(z)      # 该 Z-score 对应的概率密度高度

# 4. 代入 Delta Normal ES 公式
es_diff = sigma * pdf_z / alpha  # 风险敞口 (波动率惩罚项)
es_abs = -mu + es_diff           # 绝对 ES (取负号转化为正数亏损比例)

# 5. 格式化并导出结果
out_df = pd.DataFrame({
    'ES Absolute': [es_abs], 
    'ES Diff from Mean': [es_diff]
})

print(out_df)

### 若有多列资产，我们进行矩阵推导法
import pandas as pd
import numpy as np
from scipy.stats import norm

# 假设 df_clean 是你读取并 dropna() 后的真实数据
# df_clean = pd.read_csv("your_data.csv").dropna()

# 1. 自动监测资产数量 (获取 DataFrame 的列数)
n = df_clean.shape[1] 

# 2. 自动生成等权重数组 (如果有 5 列，就是 5 个 20%)
weights = np.array([1/n] * n)  

# 3. 提取均值向量和协方差矩阵
mu_vec = df_clean.mean().values              
cov_matrix = df_clean.cov().values           

# 4. 矩阵相乘，计算整个投资组合的均值和波动率
mu_port = weights.T @ mu_vec                 
var_port = weights.T @ cov_matrix @ weights  
sigma_port = np.sqrt(var_port)               

# 5. 计算组合 ES
alpha = 0.05
z = norm.ppf(alpha)
pdf_z = norm.pdf(z)

es_diff_port = sigma_port * pdf_z / alpha
es_abs_port = -mu_port + es_diff_port

print(f"成功监测到 {n} 个资产，已按等权重计算组合风险：")
print(f"Portfolio Volatility: {sigma_port:.4%}")
print(f"Portfolio ES (Absolute): {es_abs_port:.4%}")


### 若每列单独计算 ES，我们可以直接对每列进行循环处理
import pandas as pd
import numpy as np
from scipy.stats import norm

# 假设 df_clean 是包含多列数据的 DataFrame (比如 'AAPL', 'MSFT', 'SPY')
# df_clean = pd.read_csv("your_data.csv").dropna()

alpha = 0.05
z = norm.ppf(alpha)
pdf_z = norm.pdf(z)

# 1. 自动计算每一列的均值和标准差 (返回 Series)
mu_series = df_clean.mean()
sigma_series = df_clean.std(ddof=1)

# 2. 向量化运算！(Pandas 会自动遍历 Series 里的每一只股票代入公式)
es_diff_series = sigma_series * pdf_z / alpha
es_abs_series = -mu_series + es_diff_series

# 3. 把所有结果拼装成一张漂亮的报表
out_df_individual = pd.DataFrame({
    'ES Absolute': es_abs_series,
    'ES Diff from Mean': es_diff_series
})

print(out_df_individual)



### 计算 T分布的 ES，首先我们需要拟合出 T分布的参数（自由度 nu 和尺度 sigma），然后根据 T分布的 PDF 和 CDF 计算 ES 金额
import pandas as pd
import numpy as np
from scipy.stats import t

# 1. 读取数据 (保留所有 NA，让子弹飞一会儿)
df = pd.read_csv(r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test7_2.csv")

# =================================================================
# ⚠️ 去均值开关：如果要物理去均值，解开下面这行注释！
# df = df - df.mean()
# =================================================================

alpha = 0.05
results = {}

# 2. 遍历数据集的每一列单独作战
for col in df.columns:
    # 🎯 核心修正：局部 dropna！只剔除当前资产的空值，绝不牵连无辜！
    x = df[col].dropna().values
    
    # 防呆：如果这列全是空值，直接跳过
    if len(x) == 0:
        continue
    
    # 拟合 T 分布参数: 自由度 (df_t), 均值 (loc_t), 尺度 (scale_t)
    df_t, loc_t, scale_t = t.fit(x)
    
    # 🚨 极值保护：T分布的 ES 仅在自由度 > 1 时存在数学解析解
    if df_t <= 1.0:
        results[col] = {'ES Absolute': np.nan, 'ES Diff from Mean': np.nan, 'Warning': 'Nu <= 1, ES Infinite'}
        continue
    
    # 获取临界 t 值 (分位数) 和对应的概率密度
    t_val = t.ppf(alpha, df_t)
    pdf_t = t.pdf(t_val, df_t)
    
    # 完美的 T 分布 ES 闭式解公式
    es_diff = scale_t * (pdf_t / alpha) * ((df_t + t_val**2) / (df_t - 1))
    es_abs = -loc_t + es_diff
    
    # 将结果存入字典
    results[col] = {
        'ES Absolute': es_abs, 
        'ES Diff from Mean': es_diff,
        'Fitted Nu': df_t  # 加上自由度方便最终排错
    }

# 3. 生成优雅的 DataFrame 报表
out_df = pd.DataFrame(results).T
out_df.index.name = 'Asset'

# 格式化输出，百分比显示更具实战感
format_dict = {
    'ES Absolute': '{:.4%}', 
    'ES Diff from Mean': '{:.4%}',
    'Fitted Nu': '{:.2f}'
}

print(f"=== {len(out_df)} 列资产的 T 分布 ES 测算报告 ===")
print(out_df.style.format(format_dict, na_rep="N/A").to_string())



### 若有多列资产，我们可以在拟合 T 分布时使用多元 T 分布
import pandas as pd
import numpy as np
from scipy.stats import t

# 1. 读取多列数据并清理
df = pd.read_csv(r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test7_2.csv")
df_clean = df.dropna()

# 2. 自动监测列数，设定权重 (这里演示等权重)
n = df_clean.shape[1]
weights = np.array([1/n] * n)

# ==========================================
# 3. 【核心步骤】：多列融合成一列 (Pre-aggregation)
# DataFrame 的 .dot() 会自动把每一天的各股票收益率按权重相加
# 得到一个一维的 Pandas Series，代表你的 Portfolio 每天的真实涨跌幅
# ==========================================
portfolio_returns = df_clean.dot(weights)

# 4. 对这个组合的总收益率拟合 T 分布参数
df_t, loc_t, scale_t = t.fit(portfolio_returns)

# 5. 代入 T 分布 ES 公式计算整体风险
alpha = 0.05
t_val = t.ppf(alpha, df_t)
pdf_t = t.pdf(t_val, df_t)

es_diff_port = scale_t * (pdf_t / alpha) * ((df_t + t_val**2) / (df_t - 1))
es_abs_port = -loc_t + es_diff_port

# 6. 输出结果
out_df = pd.DataFrame({
    'Portfolio ES Absolute': [es_abs_port], 
    'Portfolio ES Diff': [es_diff_port],
    'Fitted df (nu)': [df_t]  # 顺便看看整体组合的尾巴有多肥
})

print(f"成功计算 {n} 个资产组合的整体 T 分布 ES：")
print(out_df)



### T分布下用蒙特卡洛模拟法计算 ES，这里我们先拟合出每列资产的 T 分布参数，然后根据这些参数生成随机数矩阵，最后根据置信水平计算 ES 金额
import pandas as pd
import numpy as np
from scipy.stats import t

# ==========================================
# 1. 读取并清理数据 (支持任意列数)
# ==========================================
file_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test7_2.csv"
df = pd.read_csv(file_path).dropna()

# 蒙特卡洛全局参数
np.random.seed(42)       # 固定随机种子，保证每次交作业数字一样
n_simulations = 1_000_000  # 100万次模拟
alpha = 0.05

results = {}

# ==========================================
# 2. 遍历引擎：自动计算每一列的 MC ES
# ==========================================
for col in df.columns:
    x = df[col].values
    
    # 拟合该资产的 T 分布参数
    df_t, loc_t, scale_t = t.fit(x)
    
    # 生成 100 万次模拟路径
    simulated_returns = t.rvs(df=df_t, loc=loc_t, scale=scale_t, size=n_simulations)
    
    # 找 VaR 临界点
    var_sim = np.percentile(simulated_returns, alpha * 100)
    
    # 计算 ES (切掉左侧尾巴求均值，取负号变为亏损额度)
    es_abs_sim = -np.mean(simulated_returns[simulated_returns <= var_sim])
    
    # 计算距离均值的风险敞口 (数学推导: loc_t - (-es_abs_sim) = es_abs_sim + loc_t)
    es_diff_sim = es_abs_sim + loc_t
    
    # 记录结果
    results[col] = {
        'ES Absolute': es_abs_sim, 
        'ES Diff from Mean': es_diff_sim
    }

# ==========================================
# 3. 拼装报表与导出
# ==========================================
out_df = pd.DataFrame(results).T
out_df.index.name = 'Asset'

output_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W5\testout_8.6.csv"
out_df.to_csv(output_path)

print(f"--- 成功对 {df.shape[1]} 个资产进行了各 100 万次的蒙特卡洛 ES 模拟 ---")
print(out_df)



### n列资产的 T分布蒙特卡洛 ES，这里我们先把 n 列资产每天的收益率按权重相加，得到一个组合的历史收益率序列，然后对这个组合的收益率拟合 T 分布参数，最后根据拟合的 T 分布参数生成随机数进行蒙特卡洛模拟，计算 ES 金额
import pandas as pd
import numpy as np
from scipy.stats import t

# ==========================================
# 1. 加载数据并融合为 "一维组合收益率"
# ==========================================
file_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test7_2.csv"
df = pd.read_csv(file_path).dropna()

# 自动获取资产数量，并设定权重 (此处假设等权重)
n_assets = df.shape[1]
weights = np.array([1/n_assets] * n_assets)

# 【核心动作】：矩阵点乘，将多列资产转化为 1 列组合总收益
portfolio_returns = df.dot(weights)

# ==========================================
# 2. 拟合总体的 T 分布参数
# ==========================================
df_t, loc_t, scale_t = t.fit(portfolio_returns)

# ==========================================
# 3. 蒙特卡洛引擎 (只为组合整体跑 100 万次模拟)
# ==========================================
np.random.seed(42)       
n_simulations = 1_000_000  
alpha = 0.05

# 直接生成组合的模拟收益率路径
simulated_port_returns = t.rvs(df=df_t, loc=loc_t, scale=scale_t, size=n_simulations)

# ==========================================
# 4. 计算组合的 VaR 和 ES
# ==========================================
# 找 VaR 临界点 (最差的 5% 分界线)
var_sim = np.percentile(simulated_port_returns, alpha * 100)

# 算 ES (跌穿 VaR 线的极寒数据的平均跌幅)
es_abs_sim = -np.mean(simulated_port_returns[simulated_port_returns <= var_sim])

# 算风险敞口 (剥离组合自身的预期收益)
es_diff_sim = es_abs_sim + loc_t

# ==========================================
# 5. 拼装报表与导出
# ==========================================
out_df = pd.DataFrame({
    'Portfolio ES Absolute': [es_abs_sim], 
    'Portfolio ES Diff from Mean': [es_diff_sim],
    'Portfolio Fitted Nu (df)': [df_t]  # 顺便看看你的组合尾巴有多肥
})

print(f"--- 成功融合 {n_assets} 列资产，总体蒙特卡洛 ES 模拟完成 ---")
print(out_df)



### Gaussian Copula VaR，这里我们先计算每列资产的边际分布参数（均值和标准差），然后根据这些参数把历史数据转化为标准正态分布下的概率值，接着计算这些概率值的相关系数矩阵，最后使用 Gaussian Copula 生成模拟数据并计算 VaR 金额
import pandas as pd
import numpy as np
from scipy.stats import norm, t

# ==========================================
# [外挂模块] Higham 近邻正定矩阵修复算法
# ==========================================
def project_to_positive_semidefinite(matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, 0)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

def compute_nearest_correlation_higham(target_correlation, tol=1e-9, max_iter=1000):
    correction_matrix = np.zeros_like(target_correlation)
    current_correlation = target_correlation.copy()
    for _ in range(max_iter):
        last_correlation = current_correlation.copy()
        temp_matrix = last_correlation - correction_matrix
        psd_projection = project_to_positive_semidefinite(temp_matrix)
        correction_matrix = psd_projection - temp_matrix
        
        # 强制对角线为1
        current_correlation = psd_projection.copy()
        np.fill_diagonal(current_correlation, 1.0)
        
        if np.linalg.norm(current_correlation - last_correlation, 'fro') < tol:
            break
    return current_correlation

# ==========================================
# 1. 数据加载与资产信息提取
# ==========================================
# 请确保路径正确，如果是 problem 6，请替换成对应的文件路径
portfolio = pd.read_csv(r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test9_1_portfolio.csv")
returns = pd.read_csv(r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test9_1_returns.csv")

info = {}
total_val = 0
for _, row in portfolio.iterrows():
    stock = row['Stock']
    val = row['Holding'] * row['Starting Price']
    info[stock] = {'Value': val, 'Dist': row['Distribution']}
    total_val += val

stocks = list(returns.columns)

# ==========================================
# 2. 拟合边缘分布 & 转化为均匀分布 (CDF)
# ==========================================
fits = {}
# 初始化空的 U 矩阵，确保全为 float 以兼容 NaN
u_returns = pd.DataFrame(index=returns.index, columns=stocks, dtype=float)

for stock in stocks:
    dist_type = info[stock]['Dist']
    
    # 提取纯净数据及其精确的日期索引 (避开 Pandas 维度报错地雷)
    valid_data = returns[stock].dropna()
    ret = valid_data.values
    valid_idx = valid_data.index 
    
    if dist_type == 'Normal':
        mu, std = norm.fit(ret)
        fits[stock] = {'mu': mu, 'std': std, 'type': 'Normal'}
        # 使用 .loc 按日期精准填入
        u_returns.loc[valid_idx, stock] = norm.cdf(ret, loc=mu, scale=std)
        
    elif dist_type == 'T':
        # 自由拟合，不加 floc=0，允许捕捉微小偏移
        df_t, loc_t, scale_t = t.fit(ret)
        fits[stock] = {'df': df_t, 'loc': loc_t, 'scale': scale_t, 'type': 'T'}
        u_returns.loc[valid_idx, stock] = t.cdf(ret, df=df_t, loc=loc_t, scale=scale_t)

# ==========================================
# 3. 提取 Copula 的纯粹相关性结构 & 强制正定修复
# ==========================================
# 1. 在均匀分布 U 上计算 Spearman 秩相关系数 (遇到 NA 会自动 Pairwise)
corr_matrix_df = u_returns.corr(method='spearman')

# 2. 调用 Higham 算法，强行修复可能的非正定问题 (穿上防弹衣)
corr_matrix_safe = compute_nearest_correlation_higham(corr_matrix_df.values)

# 3. 转回 DataFrame 供汇报展示
corr_matrix_safe_df = pd.DataFrame(corr_matrix_safe, index=stocks, columns=stocks)

print("\n=== Copula 中实际使用的相关系数矩阵 (Spearman Rank - 已过 Higham 修复) ===")
print(corr_matrix_safe_df)
print("-" * 65)

# ==========================================
# 4. 蒙特卡洛生成平行宇宙
# ==========================================
np.random.seed(42)  
n_simulations = 1_000_000  

# 动态生成均值向量
mean_vector = np.zeros(len(stocks))

# 生成服从我们修复后相关性结构的多元标准正态分布 Z
z_sim = np.random.multivariate_normal(mean_vector, corr_matrix_safe, n_simulations)

# 转回均匀分布 U
u_sim = norm.cdf(z_sim)

# ==========================================
# 5. 逆映射还原真实收益率 (PPF)
# ==========================================
simulated_returns = {}
for i, stock in enumerate(stocks):
    dist_type = fits[stock]['type']
    if dist_type == 'Normal':
        simulated_returns[stock] = norm.ppf(u_sim[:, i], loc=fits[stock]['mu'], scale=fits[stock]['std'])
    elif dist_type == 'T':
        simulated_returns[stock] = t.ppf(u_sim[:, i], df=fits[stock]['df'], loc=fits[stock]['loc'], scale=fits[stock]['scale'])

# ==========================================
# 6. 计算盈亏 (PnL) 与 尾部风险 (VaR/ES)
# ==========================================
pnl_sim = pd.DataFrame(simulated_returns)

# 将收益率转化为真金白银的盈亏 ($)
for stock in stocks:
    pnl_sim[stock] = pnl_sim[stock] * info[stock]['Value']

# 每日投资组合总盈亏
pnl_sim['Total'] = pnl_sim.sum(axis=1)

results = []
alpha = 0.05

for col in stocks + ['Total']:
    # PnL 变损益，亏钱为正数
    loss = -pnl_sim[col].values 
    
    # 找到最惨的 5% 临界点 (VaR)
    var = np.percentile(loss, (1 - alpha) * 100)
    
    # 计算跌穿 VaR 时的平均极端损失 (ES)
    es = np.mean(loss[loss >= var])

    # 获取底层资产的本金，用来算百分比占比
    val = info[col]['Value'] if col in info else total_val
    
    results.append({
        'Asset': col,
        'VaR95_($)': var,
        'ES95_($)': es,
        'VaR95_(%)': var / val,
        'ES95_(%)': es / val
    })

# ==========================================
# 7. 打印并导出精美风控报表
# ==========================================
results_df = pd.DataFrame(results)

# 设置金额与百分比的格式化样式
format_dict = {
    'VaR95_($)': '${:,.2f}',
    'ES95_($)': '${:,.2f}',
    'VaR95_(%)': '{:.4%}',
    'ES95_(%)': '{:.4%}'
}

print("\n=== Gaussian Copula 蒙特卡洛风控终极报告 (1,000,000 次模拟) ===")
print(results_df.style.format(format_dict).to_string())



### 计算各列PV 注意第一列是否为日期
import pandas as pd
# 1. 读取价格数据 (以第一列日期为索引)
file_path = "problem6.csv"
prices_df = pd.read_csv(file_path, index_col=0)

# 2. 获取最后一天的价格 (T时刻价格)
# .iloc[-1] 能精准切出最后一行 (即 2025年9月7日 的收盘价)
current_prices = prices_df.iloc[-1]

# 3. 填入持仓量 (Shares)
shares = 100

# 4. 计算当前市值 (PV)
# 对应元素相乘
pv = current_prices * shares

print("=== 最后一天的最新价格 (Price) ===")
print(current_prices)

print("\n=== 当前各资产独立市值 (PV) ===")
print(pv)

print(f"\n-> 总投资组合市值 (Total Portfolio Value): ${pv.sum():,.2f}")



import numpy as np
import pandas as pd
from scipy.stats import norm, t

# ==========================================
# 0. 准备测试数据 (假设你有一列收益率数据)
# ==========================================
# 这里用随机生成的含有微小肥尾的数据作为演示，
# 实际使用时替换为：data = df['Your_Asset'].dropna().values
np.random.seed(42)
data = np.random.standard_t(df=5, size=500) * 0.015 + 0.0005 

PV = 1_000_000      # 持仓市值 100万美元
n_sims = 100_000    # 模拟十万次
alpha = 0.05        # 95% 置信水平

# ==========================================
# 1. 交互式风控控制台 (Interactive Prompts)
# ==========================================
print("=" * 60)
print("🚀 欢迎使用单资产蒙特卡洛风控引擎 (MC VaR & ES)")
print("=" * 60)

ans_demean = input("⚠️ 是否需要对历史数据进行【去均值化】(Remove Mean)? (Y/N): ").strip().upper()
demean_flag = True if ans_demean == 'Y' else False

ans_dist = input("⚠️ 是否使用【T分布】刻画肥尾风险? (Y=T分布, N=正态分布): ").strip().upper()
t_dist_flag = True if ans_dist == 'Y' else False

print("\n" + "-" * 60)
print(f"⚙️ 引擎配置: 去均值化=[{'开启' if demean_flag else '关闭'}], 底层分布=[{'T分布' if t_dist_flag else '正态分布'}]")
print("-" * 60)

# ==========================================
# 2. 数据清洗与参数拟合 (Data Prep & Fitting)
# ==========================================

# A. 去均值处理 (物理平移法)
if demean_flag:
    data = data - data.mean()

# B. 分布拟合与随机数生成 (核心引擎)
if t_dist_flag:
    # 拟合 T 分布 (牢记黄金法则：去均值后也不加 floc=0，让它自由寻找微小偏移)
    df_t, loc_t, scale_t = t.fit(data)
    
    # 抽取 10万个 T分布 随机数
    sim_asset_returns = t.rvs(df_t, loc=loc_t, scale=scale_t, size=n_sims)
    
    dist_info = f"T-Distribution (Nu={df_t:.2f}, Loc={loc_t:.6f}, Scale={scale_t:.4f})"
else:
    # 拟合正态分布
    mu_norm, std_norm = norm.fit(data)
    
    # 抽取 10万个 正态分布 随机数
    sim_asset_returns = np.random.normal(loc=mu_norm, scale=std_norm, size=n_sims)
    
    dist_info = f"Normal Distribution (Mu={mu_norm:.6f}, Std={std_norm:.4f})"

# ==========================================
# 3. 计算金额盈亏与尾部风险 (PnL & VaR/ES)
# ==========================================

# 转化为金额盈亏
sim_port_pnl = PV * sim_asset_returns

# 取出最惨的 5% 临界点 (加负号变为正的亏损金额)
VaR_MC_percent = -np.percentile(sim_asset_returns, alpha * 100)
VaR_MC_dollar = -np.percentile(sim_port_pnl, alpha * 100)

# 切片：挑出所有跌破 VaR 的灾难日盈亏，算平均亏损 (ES)
tail_losses_dollar = sim_port_pnl[sim_port_pnl <= -VaR_MC_dollar]
ES_MC_dollar = -np.mean(tail_losses_dollar)

# 计算 ES 的百分比
ES_MC_percent = ES_MC_dollar / PV

# ==========================================
# 4. 打印终极风控报表
# ==========================================
print("\n" + "=" * 60)
print(f"📊 蒙特卡洛模拟分析报告 (Simulations: {n_sims:,})")
print("=" * 60)
print(f"拟合模型 : {dist_info}")
print(f"组合市值 : ${PV:,.2f}")
print(f"置信水平 : {1 - alpha:.0%}")
print("-" * 60)
print(f"🔥 MC VaR (亏损金额) : ${VaR_MC_dollar:,.2f}")
print(f"🔥 MC ES  (极端均损) : ${ES_MC_dollar:,.2f}")
print("-" * 60)
print(f"📉 MC VaR (收益率%)  : {VaR_MC_percent:.4%}")
print(f"📉 MC ES  (收益率%)  : {ES_MC_percent:.4%}")
print("=" * 60)