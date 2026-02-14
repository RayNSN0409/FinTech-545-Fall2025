import pandas as pd

### 读取数据
# 1. 设置文件路径
file_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test7_3.csv"

# 2. 读取 CSV 文件
# index_col=0 将第一列作为索引 (通常是日期列)，如果没有索引列则设置为 None
df = pd.read_csv(file_path, index_col=0)

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
        
        
        
### 拟合T分布 （即使通过JB test, 为确保计算协方差矩阵的稳定性，我们也可以拟合T分布）
import pandas as pd
import scipy.stats as stats
import numpy as np

# --- 1. 批量拟合 T分布 (MLE) ---
# 对每一列执行 fit，提取三个核心参数：自由度(Nu), 位置(Mu), 尺度(Scale)
t_params = df.apply(lambda x: pd.Series(stats.t.fit(x.dropna()), index=['Nu', 'Mu', 'Scale'])).T

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
# 关键区别：不要执行 dropna()
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
print(pd.DataFrame(B_matrix).iloc[:3, :3]) # 预览
print(pd.DataFrame(B_matrix))  # 查看全部


    
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

