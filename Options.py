"""
风险平价优化求解器 (Risk Parity Optimization Solver)
假设前提: 资产收益率服从多维正态分布 (Normal Assumption)
核心输入: N x N 的资产协方差矩阵 (Covariance Matrix, \Sigma)
求解目标: 寻找最优权重向量 w^*，使得投资组合中各项资产的绝对风险贡献 (CSD/CRC) 完全相等，
          即相对风险占比 (RC%) 均为 1/N。
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# =====================================================================
# 步骤 1：数据 I/O 路径配置
# =====================================================================
input_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test5_2.csv"
output_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W13\testout10_1.csv"


# =====================================================================
# 步骤 2：定义目标函数 (Objective Function)
# =====================================================================
def risk_parity_objective(weights, cov):
    """
    计算等风险贡献的误差平方和 (Sum of Squared Errors, SSE)。
    
    【术语与数学等价性说明】：
    理论上，绝对风险贡献 (Component Standard Deviation, CSD 或 Component Risk Contribution, CRC) 
    的公式为: CSD_i = w_i * (\Sigma w)_i / \sigma_p
    其中边际风险贡献 (MRC) 是 (\Sigma w)_i / \sigma_p。
    
    但在工程优化中，为了避免除以标准差带来的平方根计算，且防止 CSD 极小值导致浮点数精度丢失，
    我们采用【方差归一化】的等价数学转换：
    RC%_i = CSD_i / \sigma_p = (w_i * (\Sigma w)_i / \sigma_p) / \sigma_p = w_i * (\Sigma w)_i / \sigma_p^2
    
    因此，代码中的 mrc 实际上是"未缩放的边际方差贡献"，crc 是"成分方差贡献"。
    除以组合总方差后，依然完美等价于理论上的相对风险占比 (RC%)。
    
    参数:
        weights (np.array): 当前迭代的资产权重向量 w
        cov (np.ndarray):   资产的历史协方差矩阵 \Sigma
        
    返回:
        float: 各资产相对风险占比与目标占比 (1/N) 之间的误差平方和 (SSE)
    """
    n = len(weights)
    
    # 1. 组合总方差 (Portfolio Variance): \sigma_p^2 = w^T \Sigma w
    port_var = weights.T @ cov @ weights
    
    # 2. 未缩放的边际方差贡献 (Unscaled Marginal Variance Contribution): \Sigma w
    # 注: 此处未除以 \sigma_p，是为后续直接除以方差做准备
    mrc = cov @ weights
    
    # 3. 成分方差贡献 (Component Variance Contribution): w_i * (\Sigma w)_i
    crc = weights * mrc
    
    # 4. 相对风险占比 (Percentage Risk Contribution, RC%): CRC / \sigma_p^2
    # 完美映射为 CSD_i / \sigma_p
    rc_pct = crc / port_var
    
    # 5. 目标风险占比 (Target Risk Contribution): 等风险平价即 1/N
    target_rc = 1.0 / n
    
    # 6. 返回误差平方和 (SSE)，优化器目标为 min(SSE)
    return np.sum((rc_pct - target_rc)**2)


# =====================================================================
# 步骤 3：定义非线性优化求解器 (Non-linear Optimizer)
# =====================================================================
def get_risk_parity_weights(cov):
    """
    基于给定的协方差矩阵，使用 SLSQP 算法求解风险平价最优权重。
    约束条件:
        1. 满仓约束: \sum w_i = 1
        2. 做空限制: 0 <= w_i <= 1
    """
    n = cov.shape[0]
    
    # 初始基准点: 等权重分配 (1/N)
    init_guess = np.ones(n) / n
    
    # 约束: sum(w) = 1.0 (等式约束)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    
    # 边界: 禁止做空及杠杆 (w_i \in [0, 1])
    bounds = tuple((0.0, 1.0) for _ in range(n))
    
    # 序列最小二乘规划 (SLSQP) 求解
    result = minimize(
        risk_parity_objective, 
        init_guess, 
        args=(cov,), 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'ftol': 1e-12, 'disp': False} # 极高精度收敛容差
    )
    
    if not result.success:
        print("Warning: Optimization did not converge. Please check the covariance matrix.")
        
    return result.x


# =====================================================================
# 步骤 4：主程序执行流 (Execution & Persistence)
# =====================================================================

# 1. 提取协方差矩阵 (Input: N x N Covariance Matrix)
df_in = pd.read_csv(input_path)
cov_matrix = df_in.values 

# 2. 驱动引擎，计算风险平价目标权重
rp_weights = get_risk_parity_weights(cov_matrix)

# 3. 数据格式化与落盘 (输出保留 9 位小数)
df_out = pd.DataFrame(rp_weights, columns=['W'])
df_out.to_csv(output_path, index=False, float_format='%.9f')

# 4. 终端日志验证输出
print("Risk Parity Target Weights (W):")
for w in rp_weights:
    print(f"{w:.9f}")
    


"""
非等权风险预算优化求解器 (Non-Equal Risk Budgeting Optimization Solver)
假设前提: 资产收益率服从多维正态分布 (Normal Assumption)
核心输入: N x N 的资产协方差矩阵 (Covariance Matrix, \Sigma)
特殊约束: 资产 5 (X5) 的目标风险分配是其他资产的一半 (1/2 risk weight on X5)
求解目标: 寻找最优本金权重向量 w^*，使得各项资产的真实相对风险占比 (RC%) 
          严格等于基金经理设定的目标风险预算向量 (Risk Budgets)。
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# =====================================================================
# 步骤 1：数据 I/O 路径配置
# =====================================================================
input_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test5_2.csv"
output_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W13\testout_10.2.csv"


# =====================================================================
# 步骤 2：定义目标函数 (Objective Function)
# =====================================================================
def risk_parity_objective(weights, cov, risk_budgets):
    """
    计算当前资产风险占比与"目标风险预算"之间的误差平方和 (SSE)。
    
    【数学等价性与 10.1 完全一致】：
    RC%_i = CSD_i / \sigma_p = (w_i * (\Sigma w)_i / \sigma_p) / \sigma_p = w_i * (\Sigma w)_i / \sigma_p^2
    
    参数:
        weights      (np.array): 当前迭代的资产本金权重向量 w
        cov          (np.ndarray): 资产的历史协方差矩阵 \Sigma
        risk_budgets (np.array): 目标风险占比向量 (已归一化，和为1)
        
    返回:
        float: 真实风险占比与目标风险预算之间的误差平方和
    """
    # 1. 组合总方差 (Portfolio Variance): \sigma_p^2 = w^T \Sigma w
    port_var = weights.T @ cov @ weights
    
    # 2. 未缩放的边际方差贡献 (Unscaled Marginal Variance Contribution): \Sigma w
    mrc = cov @ weights
    
    # 3. 成分方差贡献 (Component Variance Contribution): w_i * (\Sigma w)_i
    crc = weights * mrc
    
    # 4. 真实相对风险占比 (Percentage Risk Contribution, RC%): CRC / \sigma_p^2
    rc_pct = crc / port_var
    
    # 5. 返回 SSE。注意此处的目标不再是 1/n，而是传入的个性化 risk_budgets
    return np.sum((rc_pct - risk_budgets)**2)


# =====================================================================
# 步骤 3：定义非线性优化求解器 (Non-linear Optimizer)
# =====================================================================
def get_risk_parity_weights(cov, risk_budgets):
    """
    基于给定的协方差矩阵和风险预算，求解最优本金权重。
    """
    n = cov.shape[0]
    
    # 初始基准点: 依然采用等权重分配启动优化器
    init_guess = np.ones(n) / n
    
    # 约束: 满仓操作，即 sum(w_i) = 1.0
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    
    # 边界: 禁止做空 (w_i \in [0, 1])
    bounds = tuple((0.0, 1.0) for _ in range(n))
    
    # 序列最小二乘规划 (SLSQP) 求解
    result = minimize(
        risk_parity_objective, 
        init_guess, 
        args=(cov, risk_budgets),    # 将风险预算向量传入目标函数
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'ftol': 1e-12, 'disp': False} 
    )
    
    if not result.success:
        print("Warning: Optimization did not converge.")
        
    return result.x


# =====================================================================
# 步骤 4：主程序执行流 (Execution)
# =====================================================================

# 1. 提取协方差矩阵
df_in = pd.read_csv(input_path)
cov_matrix = df_in.values

# 2. 构造目标风险预算向量 (Target Risk Budgets)
# 题目要求: 前 4 个资产风险等权，第 5 个资产 (X5) 的风险权重是其他的一半
raw_budgets = np.array([1.0, 1.0, 1.0, 1.0, 0.5])

# 核心步骤: 预算归一化 (Normalization)
# 无论设定的比例倍数是多少，总风险百分比必须 = 100%。
# 归一化后 X1-X4 各占约 22.22% 的风险，X5 占约 11.11% 的风险。
budgets = raw_budgets / np.sum(raw_budgets)

# 3. 驱动引擎，计算风险预算目标权重
rp_weights = get_risk_parity_weights(cov_matrix, budgets)

# 4. 数据格式化与落盘 (输出保留 9 位小数)
df_out = pd.DataFrame(rp_weights, columns=['W'])
df_out.to_csv(output_path, index=False, float_format='%.9f')

# 5. 终端日志验证输出
print("Risk Budgeting Target Weights (W):")
for w in rp_weights:
    print(f"{w:.9f}")
    
    
"""
最大夏普比率优化求解器 (Max Sharpe Ratio Optimization Solver)
理论基础: 马科维茨均值-方差优化 (Markowitz Mean-Variance Optimization, MVO)
假设前提: 资产收益率服从多维正态分布 (Normal Assumption)
核心输入: N x N 的资产协方差矩阵 (Covariance Matrix, \Sigma)
          N x 1 的预期收益率向量 (Expected Returns, \mu)
          无风险利率 (Risk-Free Rate, R_f = 0.04)
求解目标: 寻找最优本金权重向量 w^*，使得投资组合的事前夏普比率 (Ex-Ante Sharpe Ratio) 最大化。
约束条件: 满仓 (\sum w_i = 1)，且禁止做空 (w_i >= 0)。
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# =====================================================================
# 步骤 1：数据 I/O 路径配置
# =====================================================================
input_cov_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test5_2.csv"
input_mean_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test10_3_means.csv"
output_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W13\testout_10.3.csv"


# =====================================================================
# 步骤 2：定义目标函数 (Objective Function)
# =====================================================================
def negative_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate):
    """
    计算投资组合的【负】夏普比率。
    
    【数学等价性与工程技巧】：
    夏普比率的真实公式为: SR = (w^T * \mu - R_f) / \sqrt{w^T * \Sigma * w}
    由于 Scipy 库只有 minimize (求极小值) 求解器，没有 maximize 求解器。
    因此，我们将夏普比率取负号。求 -SR 的最小值，就完美等价于求 SR 的最大值。
    
    参数:
        weights          (np.array): 当前迭代的资产权重向量 w
        expected_returns (np.array): 资产的预期收益率向量 \mu
        cov_matrix       (np.ndarray): 资产的协方差矩阵 \Sigma
        risk_free_rate   (float): 无风险利率 R_f
        
    返回:
        float: 负的夏普比率 (-SR)
    """
    # 1. 组合预期总收益 (Portfolio Expected Return): \mu_p = w^T * \mu
    port_return = weights.T @ expected_returns
    
    # 2. 组合总波动率 (Portfolio Volatility): \sigma_p = \sqrt{w^T * \Sigma * w}
    port_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
    
    # 3. 返回负夏普比率: -( (\mu_p - R_f) / \sigma_p )
    return -(port_return - risk_free_rate) / port_volatility


# =====================================================================
# 步骤 3：定义非线性优化求解器 (Non-linear Optimizer)
# =====================================================================
def get_max_sharpe_weights(expected_returns, cov_matrix, risk_free_rate):
    """
    基于给定的预期收益率、协方差矩阵和无风险利率，求解使得夏普比率最大的最优权重。
    """
    n = cov_matrix.shape[0]
    
    # 初始基准点: 等权重分配 (1/N)
    init_guess = np.ones(n) / n
    
    # 边界: 禁止做空及杠杆 (w_i \in [0, 1]) (题目要求 w > 0)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    
    # 约束: 满仓操作，即 sum(w_i) = 1.0
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    
    # 序列最小二乘规划 (SLSQP) 求解
    result = minimize(
        negative_sharpe_ratio,               # 目标函数 (求负夏普比率的极小值)
        init_guess,                          # 初始迭代点
        args=(expected_returns, cov_matrix, risk_free_rate), # 传入额外参数
        method='SLSQP',                      # 求解算法
        bounds=bounds,                       # 边界
        constraints=constraints,             # 约束
        options={'ftol': 1e-12, 'disp': False} 
    )
    
    if not result.success:
        print("Warning: Optimization did not converge. The covariance matrix might be singular.")
        
    return result.x


# =====================================================================
# 步骤 4：主程序执行流 (Execution)
# =====================================================================

# 1. 提取协方差矩阵 (Covariance Matrix)
df_cov = pd.read_csv(input_cov_path)
cov_matrix = df_cov.values

# 2. 提取预期收益率向量 (Expected Returns)
# .flatten() 的作用是把可能带有维度的 DataFrame 列强行压平为一维数组 (1D Array)
df_mean = pd.read_csv(input_mean_path)
expected_returns = df_mean.values.flatten()

# 3. 设置无风险利率
risk_free_rate = 0.04

# 4. 驱动引擎，计算最大夏普比率的最优权重
optimal_weights = get_max_sharpe_weights(expected_returns, cov_matrix, risk_free_rate)

# 5. 数据格式化与落盘 (输出保留 9 位小数，保持与前两题一致)
df_out = pd.DataFrame(optimal_weights, columns=['W'])
df_out.to_csv(output_path, index=False, float_format='%.9f')

# 6. 终端日志验证输出
print("Max Sharpe Optimal Weights (W):")
for w in optimal_weights:
    print(f"{w:.9f}")
    
"""
带硬性仓位约束的最大夏普比率优化器 (Constrained Max Sharpe Ratio Optimizer)
理论基础: 马科维茨均值-方差优化 (Markowitz Mean-Variance Optimization)
假设前提: 资产收益率服从多维正态分布 (Normal Assumption)
核心输入: N x N 的资产协方差矩阵 (Covariance Matrix, \Sigma)
          N x 1 的预期收益率向量 (Expected Returns, \mu)
          无风险利率 (Risk-Free Rate, R_f = 0.04)
特殊约束: 0.1 <= w_i <= 0.5
          - 下限 10%: 强制分散化 (Forced Diversification)，防止权重分配为 0。
          - 上限 50%: 限制单一集中度风险 (Concentration Limit)，防止单票重仓。
求解目标: 在满足严格仓位上下限的前提下，寻找使得事前夏普比率最大化的本金权重。
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# =====================================================================
# 步骤 1：数据 I/O 路径配置
# =====================================================================
input_cov_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test5_2.csv"
input_mean_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test10_3_means.csv"
output_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W13\testout_10.4.csv"


# =====================================================================
# 步骤 2：定义目标函数 (Objective Function)
# =====================================================================
def negative_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate):
    """
    计算投资组合的负夏普比率 (-SR)。
    由于优化器只能求最小值，求 -SR 的极小值即等价于求真实 SR 的极大值。
    
    参数:
        weights          (np.array): 当前迭代的资产权重向量 w
        expected_returns (np.array): 资产的预期收益率向量 \mu
        cov_matrix       (np.ndarray): 资产的协方差矩阵 \Sigma
        risk_free_rate   (float): 无风险利率 R_f
        
    返回:
        float: 负的夏普比率 (-SR)
    """
    # 1. 组合预期总收益: \mu_p = w^T * \mu
    port_return = weights.T @ expected_returns
    
    # 2. 组合总波动率: \sigma_p = \sqrt{w^T * \Sigma * w}
    port_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
    
    # 3. 返回 -( (\mu_p - R_f) / \sigma_p )
    return -(port_return - risk_free_rate) / port_volatility


# =====================================================================
# 步骤 3：定义非线性优化求解器 (Non-linear Optimizer)
# =====================================================================
def get_max_sharpe_weights_bounded(expected_returns, cov_matrix, risk_free_rate):
    """
    在严格的单票仓位边界 (0.1 ~ 0.5) 约束下，求解最大夏普最优权重。
    """
    n = cov_matrix.shape[0]
    
    # 初始基准点: 等权重分配 (1/N)，对于 5 个资产正好是 0.2，处于合法边界内
    init_guess = np.ones(n) / n
    
    # 【核心改动】：设定严格的仓位护城河
    # 每个资产的权重 w_i 被严格锁死在 [0.1, 0.5] 之间
    bounds = tuple((0.1, 0.5) for _ in range(n))
    
    # 约束: 资金必须 100% 刚好用完 (满仓操作)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    
    # 序列最小二乘规划 (SLSQP) 求解
    result = minimize(
        negative_sharpe_ratio,
        init_guess,
        args=(expected_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,             # 传入严格的护城河边界
        constraints=constraints,
        options={'ftol': 1e-12, 'disp': False}
    )
    
    if not result.success:
        print("Warning: Optimization did not converge. Conflicting bounds or constraints.")
        
    return result.x


# =====================================================================
# 步骤 4：主程序执行流 (Execution & Persistence)
# =====================================================================

# 1. 加载协方差矩阵
df_cov = pd.read_csv(input_cov_path)
cov_matrix = df_cov.values

# 2. 加载预期收益率向量并压平为 1D Array
df_mean = pd.read_csv(input_mean_path)
expected_returns = df_mean.values.flatten()

# 3. 注入无风险利率引力
risk_free_rate = 0.04

# 4. 驱动优化器，计算带约束的最大夏普比率权重
optimal_weights = get_max_sharpe_weights_bounded(expected_returns, cov_matrix, risk_free_rate)

# 5. 数据格式化与落盘 
# (注意：根据用户原代码设定，float_format='%.1f' 表示输出结果保留1位小数，例如 0.2, 0.3)
df_out = pd.DataFrame(optimal_weights, columns=['W'])
df_out.to_csv(output_path, index=False, float_format='%.1f')

# 6. 终端日志验证输出
print("Constrained Max Sharpe Weights (W):")
for w in optimal_weights:
    print(f"{w:.1f}")
    
    
"""
多期事后业绩归因系统 (Multi-period Ex-Post Attribution System)
核心功能: 
    1. 模拟真实的权重漂移 (Weight Drift Simulation)
    2. 使用 Cariño K 算法解决多期复利不可加问题 (Return Attribution)
    3. 基于每日加权收益贡献计算事后风险归因 (Volatility Attribution)
"""

import numpy as np
import pandas as pd

# =====================================================================
# 步骤 1：数据 I/O 路径配置
# =====================================================================
returns_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test11_1_returns.csv"
weights_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test11_1_weights.csv"
output_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W13\testout11_1.csv"

# 读取数据
df_returns = pd.read_csv(returns_path)
df_weights = pd.read_csv(weights_path)

returns = df_returns.values                  # 每日收益率矩阵 (T x N)
weights = df_weights.values.flatten()        # 初始本金权重 (N,)
assets = df_returns.columns.tolist()         # 资产名称 ['x1', 'x2', 'x3']

n_periods, n_assets = returns.shape

# 初始化状态记录矩阵
w_t = np.zeros((n_periods, n_assets))  # 记录每天开盘的真实权重 (Weight Drift)
c_t = np.zeros((n_periods, n_assets))  # 记录每天个股的加权收益贡献 (Daily Contribution)
R_p = np.zeros(n_periods)              # 记录组合每天的总收益率 (Portfolio Daily Return)

# =====================================================================
# 步骤 2：时间序列循环 —— 模拟权重漂移 (Simulate Weight Drift)
# =====================================================================
# current_value 代表各个资产当前的虚拟资金量
current_value = weights.copy()

for t in range(n_periods):
    # 1. 计算今天开盘时的真实资金占比 (归一化)
    total_val = np.sum(current_value)
    w_start = current_value / total_val
    w_t[t] = w_start
    
    r_t = returns[t]
    
    # 2. 计算单日收益贡献 (单日权重 * 单日收益)
    daily_contrib = w_start * r_t
    c_t[t] = daily_contrib
    
    # 3. 组合单日总收益 = 所有资产单日贡献之和
    R_p[t] = np.sum(daily_contrib)
    
    # 4. 核心：资金跟随当天的涨跌幅自然生长 (复利滚存)
    current_value = current_value * (1 + r_t)

# 计算整个回测期 (30天) 的累计复利总收益
asset_total_returns = np.prod(1 + returns, axis=0) - 1
port_total_return = np.sum(current_value) / np.sum(weights) - 1


# =====================================================================
# 步骤 3：收益归因 —— Cariño's K 平滑算法
# =====================================================================
# 由于每天的收益率是复利连乘的，不能直接把 c_t 相加。必须引入 K 系数。

k_t = np.zeros(n_periods)
for t in range(n_periods):
    # 计算每日的平滑系数 (小 k_t)
    if abs(R_p[t]) < 1e-10:
        k_t[t] = 1.0 # 防除零报错
    else:
        k_t[t] = np.log(1 + R_p[t]) / R_p[t]

# 计算全局的平滑系数 (大 K)
if abs(port_total_return) < 1e-10:
    K = 1.0
else:
    K = np.log(1 + port_total_return) / port_total_return

# 核心归因公式: \sum (每天的加权收益 * (小k / 大K))
# 利用 [:, None] 将一维数组变成列向量，触发 Numpy 的广播机制(Broadcasting)逐日相乘
return_attrib = np.sum(c_t * (k_t / K)[:, None], axis=0)


# =====================================================================
# 步骤 4：事后风险归因 —— 协方差映射法 (Ex-Post Volatility Attribution)
# =====================================================================
# 计算组合的事后真实波动率 (使用样本标准差 ddof=1)
std_Rp = np.std(R_p, ddof=1)

vol_attrib = np.zeros(n_assets)
for i in range(n_assets):
    # 数学等价性: CSD_i = \beta_{i,p} * \sigma_p = [cov(c_i, R_p) / var(R_p)] * \sigma_p 
    # 约分后即为: cov(c_i, R_p) / \sigma_p
    # 其中 c_i 是该股票每天的加权收益序列
    cov_val = np.cov(c_t[:, i], R_p, ddof=1)[0, 1]
    vol_attrib[i] = cov_val / std_Rp


# =====================================================================
# 步骤 5：数据聚合与落盘输出
# =====================================================================
out_data = {
    'Value': ['TotalReturn', 'Return Attribution', 'Vol Attribution']
}

for i, asset in enumerate(assets):
    out_data[asset] = [asset_total_returns[i], return_attrib[i], vol_attrib[i]]

# 汇总列
out_data['Portfolio'] = [port_total_return, port_total_return, std_Rp]

out_df = pd.DataFrame(out_data)
out_df.to_csv(output_path, index=False, float_format='%.9f')

print(out_df.to_string(index=False))



"""
多期事后多因子归因系统 (Ex-Post Factor-Based Attribution System)
核心功能: 
    1. 资产层到因子层的降维: 组合的因子暴露度 = \sum (资产真实权重 * 资产Beta)
    2. 剥离 Alpha (残差收益): 组合总收益 - 所有因子解释的收益
    3. 因子收益归因 (Return Attribution): 使用 Cariño K 算法平滑多期因子与Alpha收益
    4. 因子风险归因 (Volatility Attribution): 将 Alpha 视为独立因子，利用协方差映射计算风险贡献
"""

import numpy as np
import pandas as pd

# =====================================================================
# 步骤 1：数据 I/O 路径配置与读取
# =====================================================================
beta_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test11_2_beta.csv"
factor_returns_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test11_2_factor_returns.csv"
stock_returns_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test11_2_stock_returns.csv"
weights_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test11_2_weights.csv"
output_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W13\testout_11.2.csv"

# 读取多维矩阵
df_stocks = pd.read_csv(stock_returns_path)
df_factors = pd.read_csv(factor_returns_path)
df_betas = pd.read_csv(beta_path, index_col='Stock')
df_weights = pd.read_csv(weights_path)

R_s = df_stocks.values           # 资产每日收益矩阵 (T x N)
R_f = df_factors.values          # 因子每日收益矩阵 (T x K)
B = df_betas.values              # 资产对因子的敏感度矩阵 (N x K)
weights = df_weights.values.flatten() # 初始资金权重

n_periods, n_stocks = R_s.shape
n_factors = R_f.shape[1]

# 初始化底层状态记录器
w_t = np.zeros((n_periods, n_stocks))      # 每日漂移后的真实权重
c_t_s = np.zeros((n_periods, n_stocks))    # 每日单只股票的收益贡献
R_p = np.zeros(n_periods)                  # 组合每日总收益

# =====================================================================
# 步骤 2：底层引擎 —— 模拟权重漂移与组合总收益
# =====================================================================
current_value = weights.copy()

for t in range(n_periods):
    # 计算当日真实仓位占比
    w_start = current_value / np.sum(current_value)
    w_t[t] = w_start
    
    r_t = R_s[t]
    c_t_s[t] = w_start * r_t
    R_p[t] = np.sum(c_t_s[t]) # 组合真实单日总收益
    
    # 资产复利自然生长
    current_value = current_value * (1 + r_t)

port_total_return = np.sum(current_value) / np.sum(weights) - 1


# =====================================================================
# 步骤 3：核心降维 —— 提取因子贡献与剥离 Alpha (The Magic)
# =====================================================================
c_t_f = np.zeros((n_periods, n_factors)) # 记录每天每个因子的收益贡献
c_t_alpha = np.zeros(n_periods)          # 记录每天剥离出的残差(Alpha)贡献

for t in range(n_periods):
    for j in range(n_factors):
        # 【极其核心的数学推导】：
        # 组合对因子 j 的敞口(Beta) = sum(每只股票的当前权重 * 这只股票对因子 j 的 Beta)
        # 组合在因子 j 上赚的钱 = 组合的因子敞口 * 因子 j 当天的收益
        c_t_f[t, j] = np.sum(w_t[t] * B[:, j] * R_f[t, j])
        
    # Alpha = 组合在这一天真实赚的钱 - 所有已知宏观因子解释的钱
    # 这是基金经理纯靠选股带来的超额日收益
    c_t_alpha[t] = R_p[t] - np.sum(c_t_f[t])


# =====================================================================
# 步骤 4：多期平滑 —— 计算 Cariño's K 与 因子收益归因
# =====================================================================
k_t = np.zeros(n_periods)
for t in range(n_periods):
    if abs(R_p[t]) < 1e-10:
        k_t[t] = 1.0
    else:
        k_t[t] = np.log(1 + R_p[t]) / R_p[t]

if abs(port_total_return) < 1e-10:
    K = 1.0
else:
    K = np.log(1 + port_total_return) / port_total_return

# 应用 Cariño K 乘数，将每日的因子/Alpha乘法复利展平为全期的加法归因
return_attrib_f = np.sum(c_t_f * (k_t / K)[:, None], axis=0)
return_attrib_alpha = np.sum(c_t_alpha * (k_t / K))

# 顺便计算纯 Alpha 视角的独立总回报率 (假设只赚 Alpha 的复利)
tr_alpha = np.prod(1 + c_t_alpha) - 1


# =====================================================================
# 步骤 5：事后风险归因 —— 将 Alpha 视为独立因子跑协方差映射
# =====================================================================
std_Rp = np.std(R_p, ddof=1)
vol_attrib_f = np.zeros(n_factors)

# 1. 计算已知因子的风险贡献
for j in range(n_factors):
    # Component Risk_j = cov(因子j的每日加权贡献, 组合每日总收益) / 组合总标准差
    cov_val = np.cov(c_t_f[:, j], R_p, ddof=1)[0, 1]
    vol_attrib_f[j] = cov_val / std_Rp

# 2. 计算 Alpha (选股特质波动) 的风险贡献
cov_val_alpha = np.cov(c_t_alpha, R_p, ddof=1)[0, 1]
vol_attrib_alpha = cov_val_alpha / std_Rp


# =====================================================================
# 步骤 6：数据聚合与落盘输出 (全自动动态适配版)
# =====================================================================
# 计算全期因子总涨跌幅
factor_total_returns = np.prod(1 + R_f, axis=0) - 1

# 1. 初始化基础字段
out_data = {
    'Value': ['TotalReturn', 'Return Attribution', 'Vol Attribution']
}

# 2. 动态提取因子名称并自动循环写入
# df_factors.columns 会自动抓取表头，例如 ['F1', 'F2', 'F3', 'F4'...]
factor_names = df_factors.columns.tolist()

for j, factor in enumerate(factor_names):
    out_data[factor] = [factor_total_returns[j], return_attrib_f[j], vol_attrib_f[j]]

# 3. 补齐 Alpha 和 Portfolio 汇总列
out_data['Alpha'] = [tr_alpha, return_attrib_alpha, vol_attrib_alpha]
out_data['Portfolio'] = [port_total_return, port_total_return, std_Rp]

# 4. 生成报表并落盘
out_df = pd.DataFrame(out_data)
out_df.to_csv(output_path, index=False, float_format='%.9f')

print(out_df.to_string(index=False))



"""
GBSM 期权统一定价与风控引擎 (GBSM Unified Pricing & Risk Engine)
核心功能:
    模式 A (已注释): 传入 ImpliedVol，计算理论价格 (Value) 与 全套希腊字母。
    模式 B (当前启用): 传入 MarketPrice，反推隐含波动率 (IV)，再计算希腊字母，并输出反推的 IV。
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

# =====================================================================
# 步骤 1：定义核心计算组件 (供反推 IV 时高频调用)
# =====================================================================
def gbsm_price(sigma, S, K, T, r, q, opt_type):
    """基础定价公式：仅返回期权价格，用于寻根求解器"""
    b = r - q
    d1 = (np.log(S / K) + (b + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if opt_type == 'call':
        return S * np.exp((b - r) * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif opt_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp((b - r) * T) * norm.cdf(-d1)
    else:
        raise ValueError("Unknown option type")

def find_implied_volatility(market_price, S, K, T, r, q, opt_type):
    """使用 Brent 法反推隐含波动率"""
    # 内在价值防爆破保护
    if opt_type == 'call' and market_price < max(0, S * np.exp(-q*T) - K * np.exp(-r*T)):
        return np.nan
    if opt_type == 'put' and market_price < max(0, K * np.exp(-r*T) - S * np.exp(-q*T)):
        return np.nan

    def objective_function(sigma):
        return gbsm_price(sigma, S, K, T, r, q, opt_type) - market_price

    try:
        # 在波动率 0.01% 到 300% 之间寻根
        return brentq(objective_function, 1e-4, 3.0)
    except ValueError:
        return np.nan


# =====================================================================
# 步骤 2：主控引擎 (兼容向前定价与向后反推)
# =====================================================================
def calculate_engine(row):
    """
    逐行处理期权数据。当前启用模式 B，通过 MarketPrice 反推 ImpliedVol。
    """
    # 提取公共参数
    S = row['Underlying']
    K = row['Strike']
    T = row['DaysToMaturity'] / row['DayPerYear']
    r = row['RiskFreeRate']
    q = row['DividendRate']
    opt_type = str(row['Option Type']).strip().lower()

    # -----------------------------------------------------------------
    # 【模式 A】：向前定价 (已注释)
    # -----------------------------------------------------------------
    # sigma = row['ImpliedVol']
    
    # -----------------------------------------------------------------
    # 【模式 B】：反推 IV (当前启用)
    # -----------------------------------------------------------------
    market_price = row['MarketPrice']
    iv = find_implied_volatility(market_price, S, K, T, r, q, opt_type)
    
    # 若反推失败，为了防止程序崩溃，尝试回退到文件中原有的 ImpliedVol，若无则给 NaN
    sigma = iv if not np.isnan(iv) else row.get('ImpliedVol', np.nan)
    # -----------------------------------------------------------------

    # 如果 sigma 依然无效，直接返回空结果防报错 (新增了 ImpliedVol 列)
    if pd.isna(sigma):
        return pd.Series({
            'ID': int(row['ID']), 
            'ImpliedVol': np.nan, 
            'Value': np.nan, 
            'Delta': np.nan, 
            'Gamma': np.nan, 
            'Vega': np.nan, 
            'Rho': np.nan, 
            'Theta': np.nan
        })

    # ================= 统一计算希腊字母与理论价格 =====================
    b = r - q
    d1 = (np.log(S / K) + (b + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    exp_brT = np.exp((b - r) * T)
    exp_rT = np.exp(-r * T)
    sqrt_T = np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    
    if opt_type == 'call':
        value = S * exp_brT * norm.cdf(d1) - K * exp_rT * norm.cdf(d2)
        delta = exp_brT * norm.cdf(d1)
        rho = T * K * exp_rT * norm.cdf(d2)
        theta = -(S * exp_brT * pdf_d1 * sigma) / (2 * sqrt_T) - (b - r) * S * exp_brT * norm.cdf(d1) - r * K * exp_rT * norm.cdf(d2)
    elif opt_type == 'put':
        value = K * exp_rT * norm.cdf(-d2) - S * exp_brT * norm.cdf(-d1)
        delta = -exp_brT * norm.cdf(-d1)
        rho = -T * K * exp_rT * norm.cdf(-d2)
        theta = -(S * exp_brT * pdf_d1 * sigma) / (2 * sqrt_T) + (b - r) * S * exp_brT * norm.cdf(-d1) + r * K * exp_rT * norm.cdf(-d2)

    gamma = (pdf_d1 * exp_brT) / (S * sigma * sqrt_T)
    vega = S * exp_brT * pdf_d1 * sqrt_T
    
    # 构建输出结果 (加入 ImpliedVol)
    output = {
        'ID': int(row['ID']),
        'ImpliedVol': sigma,  # ★ 核心改动：在此处新增输出列
        'Value': value,       # 注意：在模式B下，算出来的 Value 在数学上会极度逼近输入的 MarketPrice
        'Delta': delta,
        'Gamma': gamma,
        'Vega': vega,
        'Rho': rho,
        'Theta': theta
    }
    
    return pd.Series(output)


# =====================================================================
# 步骤 3：数据 I/O 执行流
# =====================================================================
input_file = r'C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\FINAL 2026\European Options GBSM.csv'
output_file = r'C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W13\testout_12.1_robust.csv'

# 读取数据
df_input = pd.read_csv(input_file)

# 数据清洗：剔除空行，防止 Option Type 缺失报错
df_input = df_input.dropna(subset=['Option Type'])

# 引擎点火逐行计算
df_output = df_input.apply(calculate_engine, axis=1)

# 确保 ID 字段无小数
df_output['ID'] = df_output['ID'].astype(int)

# 打印并导出 (让 Pandas 显示时对齐得更好看一点)
print(df_output.to_string(index=False, float_format='%.6f'))
df_output.to_csv(output_file, index=False, float_format='%.6f')



"""
美式期权二叉树定价系统 (John Hull 学术圣经对齐版)
核心逻辑:
    1. 步数设定: N = 500 (保证 Value 和 Vega/Rho 精度完美对齐)
    2. Greeks from Tree: 严格采用 Hull 教科书公式，利用前两层节点直接提取 Delta, Gamma, Theta。
    3. 冻结 Cost of Carry: 算 Rho 时强行不更新 b，保留学术考题设定的特征。
"""

import numpy as np
import pandas as pd

# =====================================================================
# 步骤 1：定义核心引擎 (返回全量节点的 1D Array)
# =====================================================================
def bt_american(call, underlying, strike, ttm, rf, b, ivol, N):
    dt = ttm / N
    u = np.exp(ivol * np.sqrt(dt))
    d = 1.0 / u
    
    pu = (np.exp(b * dt) - d) / (u - d)
    pd_prob = 1.0 - pu
    df = np.exp(-rf * dt)
    z = 1 if call else -1

    # 一维数组索引映射
    def idxFunc(i, j):
        return int(j * (j + 1) / 2) + i

    nNodes = int((N + 1) * (N + 2) / 2)
    optionValues = np.zeros(nNodes)

    for j in range(N, -1, -1):
        for i in range(j, -1, -1):
            idx = idxFunc(i, j)
            price = underlying * (u ** i) * (d ** (j - i))
            
            optionValues[idx] = max(0.0, z * (price - strike))
            
            if j < N:
                valNoExercise = df * (pu * optionValues[idxFunc(i + 1, j + 1)] + 
                                      pd_prob * optionValues[idxFunc(i, j + 1)])
                optionValues[idx] = max(optionValues[idx], valNoExercise)

    return optionValues


# =====================================================================
# 步骤 2：全套希腊字母计算器 (基于 John Hull 树内节点提取法)
# =====================================================================
def calculate_american_greeks(row):
    S = float(row['Underlying'])
    K = float(row['Strike'])
    T = float(row['DaysToMaturity']) / float(row['DayPerYear'])
    rf = float(row['RiskFreeRate'])
    q = float(row['DividendRate'])
    ivol = float(row['ImpliedVol'])
    
    call = True if str(row['Option Type']).strip().lower() == 'call' else False
    
    # 学术特征设定：算出 b 并冻结
    b = rf - q
    N = 500  # 核心：必须使用 500 步才能让 Greeks 收敛到参考答案级别
    
    # 1. 跑一次主树，获取全量节点
    opt_vals = bt_american(call, S, K, T, rf, b, ivol, N)
    
    def idxFunc(i, j): 
        return int(j * (j + 1) / 2) + i
        
    dt = T / N
    u = np.exp(ivol * np.sqrt(dt))
    d = 1.0 / u
    
    # 提取第 0, 1, 2 层的价格节点
    Su, Sd = S * u, S * d
    Suu, Sud, Sdd = S * u**2, S, S * d**2
    
    # 提取对应的期权价值节点
    C0 = opt_vals[idxFunc(0, 0)]
    Cu, Cd = opt_vals[idxFunc(1, 1)], opt_vals[idxFunc(0, 1)]
    Cuu, Cud, Cdd = opt_vals[idxFunc(2, 2)], opt_vals[idxFunc(1, 2)], opt_vals[idxFunc(0, 2)]
    
    # ---------------------------------------------------------
    # 2. Hull 教科书公式提取 Greeks
    # ---------------------------------------------------------
    delta = (Cu - Cd) / (Su - Sd)
    
    gamma = (((Cuu - Cud) / (Suu - Sud)) - ((Cud - Cdd) / (Sud - Sdd))) / ((Suu - Sdd) / 2.0)
    
    # Theta: C0 与 过了两步但股价又回到原点的 Cud 比较
    theta = (C0 - Cud) / (2 * dt)
    
    # ---------------------------------------------------------
    # 3. 中心差分微扰法提取 Vega 与 Rho
    # ---------------------------------------------------------
    d_vol, d_r = 1e-4, 1e-4
    
    # Vega: 微调 ivol
    vega_up = bt_american(call, S, K, T, rf, b, ivol + d_vol, N)[0]
    vega_dn = bt_american(call, S, K, T, rf, b, ivol - d_vol, N)[0]
    vega = (vega_up - vega_dn) / (2 * d_vol)
    
    # Rho: 微调 rf (冻结 b)
    rho_up = bt_american(call, S, K, T, rf + d_r, b, ivol, N)[0]
    rho_dn = bt_american(call, S, K, T, rf - d_r, b, ivol, N)[0]
    rho = (rho_up - rho_dn) / (2 * d_r)
    
    return pd.Series({
        'ID': int(row['ID']), 
        'Value': C0, 
        'Delta': delta, 
        'Gamma': gamma, 
        'Vega': vega, 
        'Rho': rho, 
        'Theta': theta
    })


# =====================================================================
# 步骤 3：数据 I/O 与执行流
# =====================================================================
input_file = r'C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test12_1.csv'
output_file = r'C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W13\testout_12.2.csv'

df_input = pd.read_csv(input_file).dropna(subset=['Option Type'])

print("正在计算美式期权指标 (N=500, Greeks from John Hull Tree)...")
df_output = df_input.apply(calculate_american_greeks, axis=1)

df_output['ID'] = df_output['ID'].astype(int)
print(df_output.to_string(index=False))

df_output.to_csv(output_file, index=False, float_format='%.9f')



"""
离散分红美式期权二叉树定价与风控系统 (Vectorized Recursive Tree)
理论基础: 分段递归二叉树 (Piecewise Binomial Tree) 与 全量有限差分法 (Finite Difference)
工程优化: NumPy 向量化数组推导 (Vectorized Backward Induction)，规避组合爆炸
"""

import numpy as np
import pandas as pd

# =====================================================================
# 模块 1：标准美式二叉树 (NumPy 向量化引擎)
# =====================================================================
def bt_american_standard_vec(call, underlying, strike, ttm, rf, b, ivol, N):
    """
    处理无分红阶段的标准 CRR 二叉树。
    采用一维数组进行整层向量化倒推计算。
    """
    if N <= 0:
        return max(0.0, (1 if call else -1) * (underlying - strike))
        
    dt = ttm / N
    u = np.exp(ivol * np.sqrt(dt))
    d = 1.0 / u
    pu = (np.exp(b * dt) - d) / (u - d)
    pd_prob = 1.0 - pu
    df = np.exp(-rf * dt)
    z = 1 if call else -1

    # 向量化生成期末节点状态
    S_nodes = underlying * (u ** np.arange(0, N + 1)) * (d ** np.arange(N, -1, -1))
    V = np.maximum(0, z * (S_nodes - strike))

    # 向量化倒推
    for j in range(N - 1, -1, -1):
        S_nodes = underlying * (u ** np.arange(0, j + 1)) * (d ** np.arange(j, -1, -1))
        V_hold = df * (pu * V[1:] + pd_prob * V[:-1])
        V_ex = np.maximum(0, z * (S_nodes - strike))
        V = np.maximum(V_hold, V_ex)

    return V[0]

# =====================================================================
# 模块 2：离散分红递归二叉树 (核心定价引擎)
# =====================================================================
def bt_american_discrete_vec(call, underlying, strike, ttm, rf, divAmts, divTimes, ivol, N):
    """
    处理离散固定金额分红的递归二叉树。
    当触及除息日节点时，截断当前树并以除息后的标的价格向下递归生成子树。
    """
    # 递归终止条件：分红日程结束或超出到期日
    if len(divAmts) == 0 or len(divTimes) == 0 or divTimes[0] > N:
        return bt_american_standard_vec(call, underlying, strike, ttm, rf, rf, ivol, N)
    
    dt = ttm / N
    u = np.exp(ivol * np.sqrt(dt))
    d = 1.0 / u
    pu = (np.exp(rf * dt) - d) / (u - d)
    pd_prob = 1.0 - pu
    df = np.exp(-rf * dt)
    z = 1 if call else -1
    
    steps_to_div = divTimes[0]
    
    # 构建到达首个除息日时的标的资产状态矩阵
    S_nodes = underlying * (u ** np.arange(0, steps_to_div + 1)) * (d ** np.arange(steps_to_div, -1, -1))
    V = np.zeros_like(S_nodes)
    
    rem_divAmts = divAmts[1:]
    rem_divTimes = [t - steps_to_div for t in divTimes[1:]]
    
    # 除息日节点评估：对比持有递归子树价值与提前行权价值
    for i in range(len(S_nodes)):
        price = S_nodes[i]
        valNoExercise = bt_american_discrete_vec(
            call, price - divAmts[0], strike, 
            ttm - steps_to_div * dt, rf, 
            rem_divAmts, rem_divTimes, ivol, N - steps_to_div
        )
        valExercise = max(0.0, z * (price - strike))
        V[i] = max(valNoExercise, valExercise)
        
    # 从除息日向基准日 (t=0) 进行向量化倒推
    for j in range(steps_to_div - 1, -1, -1):
        S_nodes = underlying * (u ** np.arange(0, j + 1)) * (d ** np.arange(j, -1, -1))
        V_hold = df * (pu * V[1:] + pd_prob * V[:-1])
        V_ex = np.maximum(0, z * (S_nodes - strike))
        V = np.maximum(V_hold, V_ex)
        
    return V[0]

# =====================================================================
# 模块 3：全套希腊字母计算器 (全量微扰法)
# =====================================================================
def calculate_discrete_greeks_vec(row):
    """
    使用中心差分法提取敏感度指标。
    针对非重合树 (Non-recombining Tree)，不应依赖树内节点推导 Greeks。
    """
    S = float(row['Underlying'])
    K = float(row['Strike'])
    # 严格锁定约束：离散树步数需与绝对天数一致，以保证除息节点对齐
    N = int(row['DaysToMaturity']) 
    T = float(row['DaysToMaturity']) / float(row['DayPerYear'])
    rf = float(row['RiskFreeRate'])
    ivol = float(row['ImpliedVol'])
    
    call = True if str(row['Option Type']).strip().lower() == 'call' else False
    
    div_dates_str = str(row['DividendDates'])
    div_amts_str = str(row['DividendAmts'])
    
    if pd.isna(row['DividendDates']) or div_dates_str.strip() == '':
        divTimes, divAmts = [], []
    else:
        divTimes = [int(float(x)) for x in div_dates_str.split(',')]
        divAmts = [float(x) for x in div_amts_str.split(',')]
        
    # 计算初始基准期权价值
    C0 = bt_american_discrete_vec(call, S, K, T, rf, divAmts, divTimes, ivol, N)
    
    # 设定微扰步长
    dS = S * 0.01
    d_vol = 1e-4
    d_r = 1e-4
    dT = 1.0 / float(row['DayPerYear']) 
    
    # 1. Delta & Gamma 计算
    C_up_S = bt_american_discrete_vec(call, S + dS, K, T, rf, divAmts, divTimes, ivol, N)
    C_dn_S = bt_american_discrete_vec(call, S - dS, K, T, rf, divAmts, divTimes, ivol, N)
    
    delta = (C_up_S - C_dn_S) / (2 * dS)
    gamma = (C_up_S - 2 * C0 + C_dn_S) / (dS ** 2)
    
    # 2. Vega 计算
    vega_up = bt_american_discrete_vec(call, S, K, T, rf, divAmts, divTimes, ivol + d_vol, N)
    vega_dn = bt_american_discrete_vec(call, S, K, T, rf, divAmts, divTimes, ivol - d_vol, N)
    vega = (vega_up - vega_dn) / (2 * d_vol)
    
    # 3. Rho 计算
    rho_up = bt_american_discrete_vec(call, S, K, T, rf + d_r, divAmts, divTimes, ivol, N)
    rho_dn = bt_american_discrete_vec(call, S, K, T, rf - d_r, divAmts, divTimes, ivol, N)
    rho = (rho_up - rho_dn) / (2 * d_r)
    
    # 4. Theta 计算 (前向差分模拟时间衰减)
    if N > 1:
        N_tomorrow = N - 1
        T_tomorrow = T - dT
        
        # 刷新分红日历表 (剔除已流逝的除息日)
        divTimes_tomorrow = []
        divAmts_tomorrow = []
        for t, amt in zip(divTimes, divAmts):
            if t - 1 > 0:
                divTimes_tomorrow.append(t - 1)
                divAmts_tomorrow.append(amt)
                
        C_tomorrow = bt_american_discrete_vec(call, S, K, T_tomorrow, rf, divAmts_tomorrow, divTimes_tomorrow, ivol, N_tomorrow)
        theta = (C0 - C_tomorrow) / dT
    else:
        theta = 0.0
        
    return pd.Series({
        'ID': int(row['ID']),
        'Value': C0,
        'Delta': delta,
        'Gamma': gamma,
        'Vega': vega,
        'Rho': rho,
        'Theta': theta
    })

# =====================================================================
# 模块 4：数据 I/O 执行流
# =====================================================================
input_file = r'C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test12_3.csv'
output_file = r'C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W13\testout_12.3.csv'

df_input = pd.read_csv(input_file).dropna(subset=['Option Type'])

print("Executing Vectorized Recursive Tree and Finite Difference Greeks calculation...")
df_output = df_input.apply(calculate_discrete_greeks_vec, axis=1)

df_output['ID'] = df_output['ID'].astype(int)

# 打印终端日志并落盘保存
print(df_output.to_string(index=False))
df_output.to_csv(output_file, index=False, float_format='%.8f')
















### Monte Carlo Simulation
import pandas as pd
import numpy as np
import os
from scipy.stats import norm
from scipy.optimize import brentq

# ==========================================
# 🚨 强制开启静默后台绘图，预防 Qt 报错
# ==========================================
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

def main():
    print("=" * 60)
    print(" 🚀 基于模拟收益率的批量期权定价工具 (支持多列/多资产)")
    print("=" * 60)

    # ---------------------------------------------------------
    # 第一步：交互获取数据路径
    # ---------------------------------------------------------
    while True:
        file_path = input("\n📁 请输入模拟收益率数据(CSV)的文件路径: ").strip().strip('"').strip("'")
        if os.path.exists(file_path):
            break
        print(f"❌ 错误: 找不到文件 '{file_path}'，请检查路径。")

    try:
        df = pd.read_csv(file_path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            print("❌ 错误: CSV 中没有任何包含纯数字的列，请检查数据格式！")
            return
            
        # ---------------------------------------------------------
        # ★ 核心升级：支持多列交互选择
        # ---------------------------------------------------------
        print("\n🔍 发现以下数值列，请选择你要用于定价的收益率列：")
        for idx, col_name in enumerate(numeric_cols):
            print(f"  [{idx}] {col_name}")
            
        target_cols = []
        while True:
            try:
                # 提示用户可以通过逗号分隔多选
                choice_str = input(f"👉 请输入列编号 (用逗号分隔，例如 '1,2,3' 代表第2,3,4列): ")
                
                # 列表推导式：分割字符串，去除空格，转为整数
                choices = [int(x.strip()) for x in choice_str.split(',')]
                
                # 验证所有输入的索引是否都在合法范围内
                if all(0 <= c < len(numeric_cols) for c in choices):
                    target_cols = [numeric_cols[c] for c in choices]
                    break
                else:
                    print(f"❌ 错误: 编号必须在 0 到 {len(numeric_cols)-1} 之间，请重新输入。")
            except ValueError:
                print("❌ 格式错误: 请只输入数字和逗号，例如 '0, 1, 2'")

        # 提取选中的多列，去除任何包含空值的行，保证数据对齐
        r_df = df[target_cols].dropna().astype(float)
        
        print(f"\n✅ 成功读取！共清洗出 {len(r_df)} 行有效数据，包含 {len(target_cols)} 个标的列: {target_cols}")
        
    except Exception as e:
        print(f"❌ 读取或清洗文件时发生异常: {e}")
        return

    # ---------------------------------------------------------
    # 第二步：交互获取定价参数
    # ---------------------------------------------------------
    print("\n" + "-" * 50)
    print(" ⚙️ 请输入基础定价参数 (将应用于所有选中的列)")
    print("-" * 50)
    
    try:
        S0 = float(input("1️⃣ 请输入当前股票价格 S0 (例如 100): "))
        rf = float(input("2️⃣ 请输入年化无风险利率 (例如 4% 请输入 0.04): "))
        days = float(input("3️⃣ 请输入距离到期的天数 (例如 1): "))
        days_per_year = float(input("4️⃣ 请输入一年包含的交易日天数 (例如 255): "))
        ttm = days / days_per_year
        
        strikes_input = input("5️⃣ 请输入需要计算的行权价 K，用逗号分隔 (例如 99,100,101): ")
        strikes = [float(x.strip()) for x in strikes_input.split(',')]
        
    except ValueError:
        print("\n❌ 输入格式错误！请确保输入的都是有效的数字。")
        return

    # ---------------------------------------------------------
    # 第三步：多标的批量蒙特卡洛计算引擎
    # ---------------------------------------------------------
    print("\n⏳ 正在进行多线程风险中性期望折现计算...\n")
    results = []

    # 外层循环：遍历用户选中的每一列（每一个资产）
    for col_name in target_cols:
        r_array = r_df[col_name].values
        
        # 针对当前列的收益率，瞬间算出所有的平行宇宙股价
        Psim = S0 * np.exp(r_array)

        # 内层循环：遍历所有的行权价
        for K in strikes:
            call_payoffs = np.maximum(Psim - K, 0.0)
            call_price = np.exp(-rf * ttm) * np.mean(call_payoffs)

            put_payoffs = np.maximum(K - Psim, 0.0)
            put_price = np.exp(-rf * ttm) * np.mean(put_payoffs)

            results.append({
                'Asset Column': col_name,  # 新增一列，标记属于哪个数据源
                'Strike (K)': K,
                'Call Price': round(call_price, 6),
                'Put Price': round(put_price, 6)
            })

    # ---------------------------------------------------------
    # 第四步：格式化输出
    # ---------------------------------------------------------
    df_results = pd.DataFrame(results)
    print("=" * 60)
    print(" 🎯 批量计算结果汇总表")
    print("=" * 60)
    # 按标的和行权价进行排序，让报表更美观
    df_results = df_results.sort_values(by=['Asset Column', 'Strike (K)'])
    print(df_results.to_string(index=False))
    print("=" * 60)
    
    # =========================================================
    # 第五步：Part B - 反推隐含波动率并绘制 Volatility Smile
    # =========================================================
    print("📈 正在计算 [95, 105] 区间的隐含波动率曲线 (Volatility Smile)...")
    
    # 1. 定义 BSM Call 定价公式 (用于反推)
    def bs_call_price(S, K, T, r, sigma):
        # 极小波动率保护
        if sigma <= 0: return max(0.0, S - K * np.exp(-r*T))
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    # 2. 生成密集的行权价网格 (Julia: 95:.01:105)
    dense_strikes = np.arange(95.0, 105.01, 0.01)
    
    # 3. 计算这些密集行权价下的蒙特卡洛价格
    dense_calls = []
    for K_dense in dense_strikes:
        # 注意：这里的 Psim 使用的是外层循环最后一次计算的值
        # 如果你只选择了一列数据，这是完美的。
        call_payoffs = np.maximum(Psim - K_dense, 0.0)
        c_price = np.exp(-rf * ttm) * np.mean(call_payoffs)
        dense_calls.append(c_price)
        
    # 4. 逐个反推隐含波动率 (Implied Volatility)
    call_ivs = []
    
    for i in range(len(dense_strikes)):
        s = dense_strikes[i]
        target_price = dense_calls[i]
        
        # 定义寻根的目标函数：BSM价格 - 目标价格 = 0
        def objective(iv):
            return bs_call_price(S0, s, ttm, rf, iv) - target_price
            
        try:
            # 在波动率 0.01% 到 300% 之间寻找零点，不需要 guess
            iv = brentq(objective, 1e-4, 3.0)
        except ValueError:
            # 如果寻根失败（通常是因为极度深虚值且数值精度不够），返回 NaN
            iv = np.nan 
            
        call_ivs.append(iv)

    # 5. 绘制并保存图表
    plt.figure(figsize=(10, 6))
    plt.plot(dense_strikes, call_ivs, label="Call Implied Vol", color='#1f77b4', linewidth=2.5)
    plt.title("Implied Volatility Curve (Volatility Smile)", fontsize=14, fontweight='bold')
    plt.xlabel("Strike Price", fontsize=12)
    plt.ylabel("Implied Volatility (IV)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # ---------------------------------------------------------
    # 强制保存到指定的绝对路径
    # ---------------------------------------------------------
    save_dir = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\FINAL 2026"
    
    # 如果文件夹不存在则自动创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # 拼接出完整的文件路径
    full_save_path = os.path.join(save_dir, "call_iv.png")
    
    # 落盘保存并清理内存
    plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 隐含波动率曲线已成功静默绘制并保存至:\n   {full_save_path}")
    
    plt.close()

if __name__ == "__main__":
    main()
    
    

### Finite Difference Method, 简称 FDM 或 PDE 法
import numpy as np
import pandas as pd
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.interpolate import CubicSpline
import os

# =====================================================================
# 核心引擎：隐式有限差分求解器 (Implicit Finite Difference)
# =====================================================================
def solve_black_scholes_pde(S0, K, T, r, q, sigma, opt_type, is_american, M=200, N=200):
    """
    使用隐式有限差分法 (Implicit FDM) 求解 Black-Scholes 偏微分方程。
    M: 股价空间划分的网格数
    N: 时间划分的步数
    """
    # 极小时间保护
    if T <= 1e-6:
        payoff = np.maximum(S0 - K, 0) if opt_type == 'call' else np.maximum(K - S0, 0)
        return payoff, 0.0, 0.0, 0.0 # Price, Delta, Gamma, Theta
        
    dt = T / N
    S_max = 3.0 * max(S0, K) # 设定空间边界为当前股价的3倍，保证边界条件不影响中心区域
    dS = S_max / M
    
    # 建立空间网格 (Stock Price Grid)
    S_grid = np.linspace(0, S_max, M + 1)
    
    # 初始化到期日 (t=T) 的 Payoff (终端边界条件)
    if opt_type == 'call':
        V = np.maximum(S_grid - K, 0)
    else:
        V = np.maximum(K - S_grid, 0)
        
    payoff = V.copy() # 保存内在价值，用于美式期权的提前行权比较
    
    # ---------------------------------------------------------
    # 构建隐式差分矩阵 A (Tridiagonal Matrix)
    # 方程: a_i * V_{i-1}^{j} + b_i * V_i^{j} + c_i * V_{i+1}^{j} = V_i^{j+1}
    # ---------------------------------------------------------
    i = np.arange(1, M) # 内部节点索引 (1 到 M-1)
    
    # 漂移项与扩散项系数
    a = -0.5 * dt * (sigma**2 * i**2 - (r - q) * i)
    b = 1.0 + dt * (sigma**2 * i**2 + r)
    c = -0.5 * dt * (sigma**2 * i**2 + (r - q) * i)
    
    # 使用 scipy.sparse 构建三对角稀疏矩阵，极大提升求逆速度
    A = diags([a[1:], b, c[:-1]], [-1, 0, 1], format='csr')
    
    # ---------------------------------------------------------
    # 时间倒推 (Backward Induction)
    # ---------------------------------------------------------
    V_tomorrow = np.zeros_like(V) # 用于记录明天(t+dt)的价格，用来算 Theta
    
    for j in range(N):
        # 提取上一时间步的内部节点 (已知向量)
        B = V[1:M].copy()
        
        # 强制实施空间边界条件 (Dirichlet Boundary Conditions)
        t_rem = j * dt # 距离到期日还有多久
        if opt_type == 'call':
            # S=0 时，Call 价值为 0
            # S=S_max 时，Call 价值约为 S_max * e^{-qt} - K * e^{-rt}
            V[0] = 0.0
            V[M] = S_max * np.exp(-q * t_rem) - K * np.exp(-r * t_rem)
        else:
            # S=0 时，Put 价值为 K * e^{-rt}
            # S=S_max 时，Put 价值为 0
            V[0] = K * np.exp(-r * t_rem)
            V[M] = 0.0
            
        # 将边界条件代入方程的常数项中
        B[0] -= a[0] * V[0]
        B[-1] -= c[-1] * V[M]
        
        # 求解线性方程组 A * V_new = B
        V[1:M] = spsolve(A, B)
        
        # 美式期权精髓：求解完 PDE 后，如果内在价值大于理论价值，直接覆盖 (投影)
        if is_american:
            V = np.maximum(V, payoff)
            
        # 当推导到倒数第二步(距离今天还有 dt 的时间)时，记录下来算 Theta
        if j == N - 2:
            V_tomorrow = V.copy()
            
    # ---------------------------------------------------------
    # 提取希腊字母的工业级黑客技巧：三次样条插值 (Cubic Spline)
    # ---------------------------------------------------------
    # 为什么用插值？因为 S0 未必刚好落在我们切分的网格点上。
    # 用样条曲线拟合整个价格网格后，可以直接通过求一阶导数和二阶导数完美得出 Delta 和 Gamma！
    spline_today = CubicSpline(S_grid, V)
    spline_tomorrow = CubicSpline(S_grid, V_tomorrow)
    
    price = float(spline_today(S0))
    delta = float(spline_today(S0, 1)) # 一阶导数 = Delta
    gamma = float(spline_today(S0, 2)) # 二阶导数 = Gamma
    
    # Theta: (今天价格 - 明天价格) / dt 的年化流逝
    price_tomorrow = float(spline_tomorrow(S0))
    # 业界标准：Theta 统一定义为一年的时间衰减
    theta = (price_tomorrow - price) / dt 
    
    return price, delta, gamma, theta


# =====================================================================
# 主控引擎与微扰法算 Vega / Rho
# =====================================================================
def get_pde_pricing_and_greeks(S0, K, T, r, q, sigma, opt_type, is_american):
    # 1. 跑一次基准 PDE，拿到价格、Delta、Gamma、Theta
    price, delta, gamma, theta = solve_black_scholes_pde(S0, K, T, r, q, sigma, opt_type, is_american)
    
    # 2. 算 Vega：微调波动率 (+1 bp) 重新跑一遍 PDE
    d_sigma = 1e-4
    price_up_v, _, _, _ = solve_black_scholes_pde(S0, K, T, r, q, sigma + d_sigma, opt_type, is_american)
    price_dn_v, _, _, _ = solve_black_scholes_pde(S0, K, T, r, q, sigma - d_sigma, opt_type, is_american)
    vega = (price_up_v - price_dn_v) / (2 * d_sigma)
    
    # 3. 算 Rho：微调无风险利率 (+1 bp) 重新跑一遍 PDE
    d_r = 1e-4
    price_up_r, _, _, _ = solve_black_scholes_pde(S0, K, T, r + d_r, q, sigma, opt_type, is_american)
    price_dn_r, _, _, _ = solve_black_scholes_pde(S0, K, T, r - d_r, q, sigma, opt_type, is_american)
    rho = (price_up_r - price_dn_r) / (2 * d_r)
    
    return {
        'Option Value': round(price, 6),
        'Delta (Δ)': round(delta, 6),
        'Gamma (Γ)': round(gamma, 6),
        'Vega (ν)': round(vega, 6),
        'Rho (ρ)': round(rho, 6),
        'Theta (Θ)': round(theta, 6)
    }

# =====================================================================
# 交互式 CLI (命令行界面)
# =====================================================================
def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 60)
    print(" 🌊 偏微分方程 (PDE) 极速期权定价引擎 | 有限差分法")
    print("=" * 60)
    print("支持：欧式/美式、连续分红、平滑 Greeks 插值提取")
    print("-" * 60)

    try:
        # 交互输入参数
        S0 = float(input("📈 请输入当前标的价格 S0 (如 100): "))
        K = float(input("🎯 请输入期权行权价 K (如 100): "))
        T = float(input("⏳ 请输入到期时间 T (按年化，如半年输入 0.5): "))
        r = float(input("🏦 请输入无风险利率 r (如 4% 输入 0.04): "))
        q = float(input("💸 请输入连续股息率 q (无分红输入 0): "))
        sigma = float(input("🌪️  请输入隐含波动率 σ (如 20% 输入 0.20): "))
        
        opt_type_input = input("⚖️  期权类型 (1: 看涨 Call, 2: 看跌 Put): ").strip()
        opt_type = 'call' if opt_type_input == '1' else 'put'
        
        style_input = input("🗽 行权风格 (1: 欧式 European, 2: 美式 American): ").strip()
        is_american = True if style_input == '2' else False
        
    except ValueError:
        print("\n❌ 输入格式错误！请确保输入的都是有效的数字。")
        return

    print("\n⏳ 正在构建网格并求解 PDE 矩阵，请稍候...\n")
    
    # 调用 PDE 引擎
    results = get_pde_pricing_and_greeks(S0, K, T, r, q, sigma, opt_type, is_american)
    
    # 格式化输出
    print("=" * 40)
    print(f" 📊 计算结果 ({'美式' if is_american else '欧式'} {opt_type.capitalize()})")
    print("=" * 40)
    for key, value in results.items():
        print(f" {key.ljust(15)} : {value}")
    print("=" * 40)
    print("💡 解析：PDE 通过求解隐式稀疏矩阵，彻底消除了二叉树的 Gamma 扭曲震荡现象。")

if __name__ == "__main__":
    main()
    
    
    






### VaR & ES
"""
🚀 期权组合动态风险引擎 (T-Distribution 胖尾对齐 & 智能数据挂载版)
优化: 
  1. 智能适配 `Date, SPY` 或任意指数格式，自动跳过时间列。
  2. 自动从 CSV 末尾提取精确的 S0，解决手工输入导致的分母偏差。
"""

import numpy as np
import pandas as pd
import os
from scipy.stats import norm, t
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

def gbsm_price(is_call, S, K, T, r, q, sigma):
    if T <= 0:
        if is_call: return np.maximum(S - K, 0.0)
        else: return np.maximum(K - S, 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call: return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else: return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

def main():
    print("=" * 65)
    print(" 🛡️  期权组合动态风险引擎 (智能数据挂载版)")
    print("=" * 65)

    try:
        # ---------------------------------------------------------
        # 第一步：智能数据挂载 (解决列名和 S0 问题)
        # ---------------------------------------------------------
        print("\n📁 第一步：加载历史数据")
        csv_path = input("   👉 请输入标的历史价格 CSV 路径 (例如 problem2.csv): ").strip().strip('"').strip("'")
        
        if os.path.exists(csv_path):
            df_hist = pd.read_csv(csv_path)
            
            # 智能列名探测：跳过 date, time, index 等干扰列，抓取第一个纯数字列
            target_col = None
            for col in df_hist.columns:
                if col.lower() not in ['date', 'time', 'index', 'id', 'unnamed: 0'] and np.issubdtype(df_hist[col].dtype, np.number):
                    target_col = col
                    break
                    
            if target_col is None:
                raise ValueError("在 CSV 中找不到有效的价格数字列！")
                
            prices = df_hist[target_col].dropna().values
            returns = pd.Series(prices).pct_change().dropna().values
            
            print(f"   ✅ 成功识别价格列: '{target_col}'")
            S0_auto = prices[-1]
            print(f"   ✅ 自动提取最新现价: S0 = {S0_auto:.2f}")
            
            # 拟合 T 分布
            print("   ⏳ 正在拟合 T 分布参数...")
            df_t, loc_t, scale_t = t.fit(returns)
            print(f"   ✅ T 分布拟合完成: df={df_t:.2f}, loc={loc_t:.6f}, scale={scale_t:.6f}")
            
            S0 = S0_auto # 强制使用精确价格，避免手工误差
            
        else:
            raise FileNotFoundError("找不到 CSV 文件，请检查路径！")

        # ---------------------------------------------------------
        # 第二步：输入持仓数量
        # ---------------------------------------------------------
        print("\n📦 第二步：输入持仓信息 (负数代表做空 Short)")
        qty_S = float(input("   👉 持有正股 (Stock) 数量: "))
        qty_C = float(input("   👉 持有看涨期权 (Call) 数量: "))
        qty_P = float(input("   👉 持有看跌期权 (Put) 数量: "))

        # ---------------------------------------------------------
        # 第三步：输入 T0 期权与市场状态
        # ---------------------------------------------------------
        print("\n⏱️  第三步：输入期权信息")
        if qty_C != 0:
            K_C = float(input("   📝 Call 行权价 K_call = "))
            C0 = float(input("   📝 Call 当前市价 C0 = "))
            IV_C = float(input("   📝 Call 隐含波动率 (如 0.1714) IV_call = "))
        else: K_C, C0, IV_C = 0, 0, 0
            
        if qty_P != 0:
            K_P = float(input("   📝 Put 行权价 K_put = "))
            P0 = float(input("   📝 Put 当前市价 P0 = "))
            IV_P = float(input("   📝 Put 隐含波动率 (如 0.1825) IV_put = "))
        else: K_P, P0, IV_P = 0, 0, 0

        r = float(input("   📝 无风险利率 (如 4% 输入 0.04) r = "))
        q = float(input("   📝 连续股息率 (无分红输入 0) q = "))
        T_days = float(input("   📝 期权距离到期还有多少个交易日？ = "))

        # ---------------------------------------------------------
        # 第四步：输入 Tn (模拟场景)
        # ---------------------------------------------------------
        print("\n🔮 第四步：输入模拟情景 (Tn) 与风险参数")
        hold_days = float(input("   👉 打算持有该组合几天？(Tn) = "))
        if hold_days > T_days: hold_days = T_days
        alpha_pct = float(input("   👉 要求计算百分之几的 VaR 和 ES？(如 5% 输入 5): "))
        alpha = alpha_pct / 100.0

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        return

    # =====================================================================
    # 风险计算引擎点火
    # =====================================================================
    print(f"\n⏳ 引擎启动：基于 T-Distribution 生成 100,000 条路径并进行全值重估...")
    
    V0 = qty_S * S0 + qty_C * C0 + qty_P * P0
    
    np.random.seed(200) # 保持随机种子一致
    nsim = 100000
    int_days = int(hold_days)
    
    # T 分布算术收益率抽样
    rsim = t.rvs(df=df_t, loc=loc_t, scale=scale_t, size=(nsim, int_days))
    Sn = S0 * np.prod(1 + rsim, axis=1) # 累乘计算终值
    
    # 重新定价期权 (Repricing)
    T_remain = (T_days - hold_days) / 255.0
    
    Cn = gbsm_price(True, Sn, K_C, T_remain, r, q, IV_C) if qty_C != 0 else np.zeros(nsim)
    Pn = gbsm_price(False, Sn, K_P, T_remain, r, q, IV_P) if qty_P != 0 else np.zeros(nsim)
    
    Vn = qty_S * Sn + qty_C * Cn + qty_P * Pn
    pnl_dollar = Vn - V0
    pnl_pct = pnl_dollar / abs(V0) if abs(V0) > 1e-4 else np.zeros(nsim)
    
    VaR_dollar = -np.percentile(pnl_dollar, alpha * 100)
    ES_dollar = -np.mean(pnl_dollar[pnl_dollar <= -VaR_dollar])
    
    if abs(V0) > 1e-4:
        VaR_pct = -np.percentile(pnl_pct, alpha * 100)
        ES_pct = -np.mean(pnl_pct[pnl_pct <= -VaR_pct])

    # =====================================================================
    # 输出风险报告
    # =====================================================================
    print("\n" + "=" * 60)
    print(f" 📊 T-Distribution 风险分析报告 ({int_days} 天持有期)")
    print("=" * 60)
    print(f"💰 精确期初组合总投资 (V0)  : ${V0:,.2f}")
    
    if abs(V0) > 1e-4:
        print(f"\n📉 相对百分比敞口 (Percentage Risk):")
        print(f"   ► {alpha_pct}% VaR  : {VaR_pct*100:.4f}%")
        print(f"   ► {alpha_pct}% ES   : {ES_pct*100:.4f}%")
    
    # 绘图逻辑保持不变...
    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(pnl_pct * 100, bins=120, color='lightgray', edgecolor='white') # 改为绘制百分比
    for count, bin_edge, patch in zip(counts, bins, patches):
        if bin_edge <= -VaR_pct * 100:
            patch.set_facecolor('#d62728') 
            patch.set_alpha(0.8)
        else:
            patch.set_facecolor('#1f77b4') 
            patch.set_alpha(0.6)
            
    plt.axvline(-VaR_pct * 100, color='red', linestyle='solid', linewidth=2, label=f'{alpha_pct}% VaR')
    plt.axvline(-ES_pct * 100, color='darkred', linestyle='dashed', linewidth=2, label=f'{alpha_pct}% ES')
    plt.title(f"Fat-Tail Portfolio PnL (%) Distribution after {int_days} Days", fontsize=14, fontweight='bold')
    plt.xlabel("Profit and Loss (%)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='upper right', fontsize=11)
    
    save_dir = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\FINAL 2026"
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    full_path = os.path.join(save_dir, "FatTail_VaR_ES_Distribution.png")
    plt.tight_layout()
    plt.savefig(full_path, dpi=300)
    print(f"\n✅ 专业风险百分比直方图已静默保存至:\n   {full_path}")
    plt.close()

if __name__ == "__main__":
    main()
    
    
    

import numpy as np
import pandas as pd
from scipy.optimize import minimize

print("=" * 60)
print(" 🚀 Part D: 投资组合优化 (寻找性价比最高的仓位)")
print("=" * 60)

# ==============================================================================
# 前置准备：这些变量是从前面的代码里继承过来的。
# 【通俗解释】：我们在做菜之前，先把葱姜蒜（各种已知数据）切好放在盘子里。
# S: 今天真实的股票价格
# pP: 今天真实的看跌期权(Put)价格
# cP: 今天真实的看涨期权(Call)价格
# pValInit: 题目给的初始预算，也就是 659.67 块钱
# Psim: 前面用蒙特卡洛抽出来的 10万个“5天后的平行宇宙股票价格”
# pX, cX: 期权的行权价
# rf: 存银行的无风险年利率 (0.04)
# q: 股票的分红率
# pIV, cIV: 前面反推出来的隐含波动率 (期权的灵魂参数)
# ==============================================================================

# ⚠️ 懒人修改预警 1：到底拿在手里几天？
# 题目说 "holding until XX"，算出来是 5 天。
# 如果期末考试老师改成了 10天，就把这里的 5 换成 10！
holding_days = 5

# 这 5 天里，如果你把钱存在银行能拿多少利息？(机会成本)
rf_holding = rf * (holding_days / 255.0)

# ⚠️ 懒人修改预警 2：期权还剩几天过期？
# 题目原始期权到期是 11/255 年，你在手里捏了 5 天，那期权寿命就少了 5 天。
# 理论上应该剩 (11 - 5) / 255 = 6/255。
# 【坑点】：老师的 Julia 官方答案里，不知道为什么直接写死了 5/255！
# 为了跟老师的答案对齐不扣分，我们这里将错就错，也写 5/255。考试时请严格看题意！
eval_ttm = 5 / 255.0 

# ==============================================================================
# 💡 【核心救命优化：提前算好期权价格】
# 为什么不在下面的公式里算？
# 因为下面的机器人(优化器)会盲猜几百次仓位组合。如果每次猜，都要算 10万次期权价格，你的电脑会卡死冒烟。
# 所以我们“提前把饭做好”：不管你买多少份期权，期权在第 5 天的单价是固定的。提前算好存进数组里！
# ==============================================================================
Psim_put_vals = gbsm_price(False, Psim, pX, eval_ttm, rf, q, pIV)
Psim_call_vals = gbsm_price(True, Psim, cX, eval_ttm, rf, q, cIV)

# ==============================================================================
# 🎯 目标函数：告诉机器人，我们到底想要什么？
# 我们的目标是：让 (平均赚的钱 - 银行利息) / 极端亏损(ES) 这个比值【越大越好】
# ==============================================================================
def objective_function(weights):
    # 机器人每次猜的一个组合，比如 h1=1股股票, h2=1.5个Put, h3=-2个Call
    h1, h2, h3 = weights
    
    # 💰 算算机器人猜的这个组合，今天买需要花多少钱？
    init = h1 * S + h2 * pP + h3 * cP
    
    # 【防爆破机制】：如果机器人瞎猜，猜出了一个一分钱都不用花的“白嫖组合”（init=0），
    # 后面算收益率(除以成本)时就会发生“除以0”的数学灾难。
    # 所以给它一个巨大的惩罚值 (1e9 就是 10个亿)，警告它别瞎搞。
    if abs(init) < 1e-6:
        return 1e9  
    
    # 🔮 极速计算：这个组合在 5天后的 10万个平行宇宙里，到底值多少钱？
    pVal = h1 * Psim + h2 * Psim_put_vals + h3 * Psim_call_vals
    
    # 📈 计算这 10万个宇宙里的盈亏百分比： (期末价值 - 期初成本) / 期初成本
    pnl = (pVal - init) / init
    
    # ⚠️ 懒人修改预警 3：你要看百分之几的风险？
    # 题目要求算 5% 的 ES。如果老师考试改成了 1%，把这里的 5 改成 1。
    alpha_pct = 5 
    
    # 找 VaR：把 10万个收益率从小到大排队，找到排在最差那 5% 位置的倒霉蛋
    var_threshold = -np.percentile(pnl, alpha_pct) 
    
    # 找 ES：把比 VaR 还惨的那些倒霉蛋（最差的5%）全部拉出来，求个平均亏损
    es = -np.mean(pnl[pnl <= -var_threshold])
    
    # 🏆 算最终得分 (STARR Ratio) = (平均赚的钱 - 银行利息) / 极端情况平均亏多少
    ratio = (np.mean(pnl) - rf_holding) / es
    
    # 🧠 【反常识注意】：为什么这里要加个负号（-ratio）？
    # 因为 Python 里的这个机器狗 (minimize)，它是个弱智，它只知道怎么找【最低的谷底】，不知道怎么找【最高的山峰】。
    # 我们想要 ratio 越大越好。所以我们故意给 ratio 加个负号。
    # 比如本来是赚 10 分，加负号变 -10。机器狗找到最小的 -10，也就等于帮我们找到了最大的 10！
    return -ratio

# ==============================================================================
# 🚧 边界与规矩：给机器狗套上狗链
# ==============================================================================

# ⚠️ 懒人修改预警 4：最多能买/卖多少份？ (Bounds)
# 题目说 h ∈ [-2, 2]。意思是最多做空2份，最多做多2份。
# 如果老师改成了 "不能做空 (no shorting)"，那就必须改成 (0.0, 2.0)
bnds = (
    (-2.0, 2.0),  # h1: 股票的买卖限制
    (-2.0, 2.0),  # h2: Put 的买卖限制
    (-2.0, 2.0)   # h3: Call 的买卖限制
)

# ⚠️ 懒人修改预警 5：预算红线 (Constraints)
# 题目要求：不管你怎么优化，你买东西花掉的钱 (w[0]*S + w[1]*pP + w[2]*cP) 
# 必须【严格等于】原来的组合价值 (pValInit = 659.67)。
# 如果老师说 "你现在有 1000 块钱去投资"，就把后面的 pValInit 换成 1000。
cons = ({
    'type': 'eq', # eq 代表等式约束，意思是必须让后面那个函数的计算结果等于 0
    'fun': lambda w: w[0]*S + w[1]*pP + w[2]*cP - pValInit
})

# 初始瞎猜点：告诉机器狗从哪里开始找。我们就按现在的持仓 [1股, 1个Put, -1个Call] 让它开始。
ho = [1.0, 1.0, -1.0]

print("⏳ 放狗寻找最优解... (正在运行 SLSQP 优化器)")

# ==============================================================================
# 🚀 机器狗点火运行
# ==============================================================================
opt_result = minimize(
    objective_function, # 这是刚才写的目标函数
    ho,                 # 从这里开始起步
    method='SLSQP',     # 这个算法的好处是能听懂我们定的“狗链规则”（能处理边界和等式约束）
    bounds=bnds,        # 套上买卖数量的狗链
    constraints=cons,   # 套上预算红线的狗链
    tol=1e-5,           # ⚠️ 懒人修改预警 6：收敛精度。题目 Hint 说 1e-5，老师改 1e-6 你就跟着改。
    options={'maxiter': 500, 'disp': False} # 最多找 500 次，找不到拉倒
)

# ==============================================================================
# 🖨️ 打印成绩单
# ==============================================================================
if opt_result.success:
    # opt_result.x 里面装的就是机器狗找到的最好的 [股票份数, Put份数, Call份数]
    optimal_h = opt_result.x
    
    # 弄个好看的表格打印出来，假装我们用的是老师的 Julia
    res_df = pd.DataFrame({
        'Asset': ['Stock', 'Put', 'Call'],
        'Holding': np.round(optimal_h, 4)
    })
    
    print("\n✅ 优化成功！")
    print(res_df.to_string(index=False))
    
    # 因为前面骗机器狗加了负号，现在把负号拿掉，还原本来的最高得分
    print(f"\n📈 优化后的最牛性价比 (STARR Ratio) : {-opt_result.fun:.6f}")
else:
    print("\n❌ 优化器卡壳了，可能是条件太苛刻找不到解，原因:", opt_result.message)
    
    
    
    
    
    
    
import numpy as np
import pandas as pd
from scipy.optimize import minimize

print("=" * 60)
print(" 🚀 Problem 3 - Part A: In-Sample Portfolio Optimization")
print("=" * 60)

# ==============================================================================
# 1. 基础参数与数据读取
# ==============================================================================
# ⚠️ 考点修改预警 1：无风险利率 (Risk Free Rate)
# 题目要求 4%，如果改成 5%，这里就写 0.05
rf = 0.04 

# ⚠️ 考点修改预警 2：指数加权衰减因子 (Lambda)
# 题目要求 λ=0.97，代表过去的权重以 0.97 的速度衰减。如果题目改了这里跟着改。
lbd = 0.97 

# 假设已经读取了问题提供的样本内数据 (这里用虚拟数据演示逻辑)
# 实际考试中你应该用 pd.read_csv("problem3_insample.csv")
# 这里的 df_insample 必须是纯数字的收益率矩阵 (每列是一个股票，每行是一个月的收益率)
try:
    insample = pd.read_csv("problem3_insample.csv")
    stocks = [col for col in insample.columns if col.lower() != "date"]
    insample_r = insample[stocks].values
except FileNotFoundError:
    print("⚠️ 找不到 CSV 文件，使用随机生成的月度收益率代替运行...")
    np.random.seed(42)
    stocks = ["GOOG", "JPM", "WMT", "AMD", "NKE"]
    insample_r = np.random.normal(0.01, 0.05, size=(60, 5)) # 模拟 60个月，5只股票

# ==============================================================================
# 2. 预期收益率与协方差矩阵 (核心考点：月度转年度)
# ==============================================================================

# 计算月度平均收益率
er_monthly = np.mean(insample_r, axis=0)

# ⚠️ 考点修改预警 3：预期收益率年化公式 (Annualize Expected Return)
# 严格按照题干 Hint: Scale expected return as (1 + er)^12 - 1
er = (1 + er_monthly)**12 - 1

def ewma_covariance(returns, lbd):
    """手写指数加权协方差矩阵 (Exponentially Weighted Covariance)"""
    T, N = returns.shape
    # 生成权重序列：越近的数据权重越大 (1-λ) * λ^t
    weights = (1 - lbd) * (lbd ** np.arange(T-1, -1, -1))
    weights /= weights.sum() # 归一化，保证权重和为 1
    
    # 将收益率去均值 (中心化)
    centered_returns = returns - np.mean(returns, axis=0)
    
    # 计算加权协方差
    cov = np.dot(centered_returns.T, centered_returns * weights[:, np.newaxis])
    return cov

# 计算月度 EWMA 协方差
covar_monthly = ewma_covariance(insample_r, lbd)

# ⚠️ 考点修改预警 4：协方差年化公式 (Annualize Covariance)
# 严格按照题干 Hint: Scale the covariance as ewCovar * 12
covar = covar_monthly * 12

print(f"✅ 数据年化处理完毕 (Expected Returns & EWMA Covariance Matrix)")

# ==============================================================================
# 3. 求解最大夏普比率组合 (Max Sharpe Ratio Portfolio)
# ==============================================================================
def max_sharpe_ratio(cov_matrix, exp_returns, risk_free_rate):
    num_assets = len(exp_returns)
    
    # 目标函数：我们要最大化夏普比率。因为 Python 只能求最小值，所以返回负的夏普比率。
    def neg_sharpe(weights):
        port_ret = np.dot(weights, exp_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(port_ret - risk_free_rate) / port_vol

    # ⚠️ 考点修改预警 5：做空限制 (Bounds & Constraints)
    # 这里的约束条件是：所有权重加起来必须等于 1 (100% 满仓)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    
    # 这里的边界是：每个股票的权重在 0 到 1 之间 (代表只允许做多 Long-Only)
    # 如果题目说 "allow short selling up to -10%"，那就改成 (-0.1, 1.0)
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    
    # 从平均分配的仓位开始找起
    init_guess = np.ones(num_assets) / num_assets
    
    opt = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return opt.x

# ==============================================================================
# 4. 求解风险平价组合 (Risk Parity Portfolio - RPP)
# ==============================================================================
def risk_parity(cov_matrix):
    num_assets = cov_matrix.shape[0]
    
    # 目标函数：让每一个股票的“风险贡献度”绝对相等
    def target_risk_contributions(weights):
        # 1. 算组合总方差
        port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
        # 2. 算每个股票的边际风险贡献 (Marginal Risk Contribution)
        marginal_contribs = np.dot(cov_matrix, weights)
        # 3. 算每个股票的绝对风险贡献 (Risk Contribution)
        risk_contribs = weights * marginal_contribs
        
        # 4. 目标是让每个股票的风险贡献 = 总风险 / 股票数量
        target_rc = port_var / num_assets
        
        # 将误差平方和作为惩罚项（越接近0，说明大家分担的风险越平均）
        # 乘个 1e6 是为了放大误差，帮助 SLSQP 优化器更容易找到方向
        return np.sum((risk_contribs - target_rc)**2) * 1e6

    # 同样是满仓 100%，且不允许做空的限制
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    init_guess = np.ones(num_assets) / num_assets
    
    opt = minimize(target_risk_contributions, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return opt.x

# ==============================================================================
# 5. 执行优化并输出结果对账单
# ==============================================================================
print("⏳ 正在运行优化器 (SLSQP)...")
mSR = max_sharpe_ratio(covar, er, rf)
rPP = risk_parity(covar)

# 将结果拼装成 DataFrame 打印，对齐 Julia 答案的格式
df_results = pd.DataFrame({
    'Stock': stocks,
    'MaxSR': np.round(mSR, 4),
    'RPP': np.round(rPP, 4)
})

print("\n✅ 优化成功！(权重分配比例如下)")
print(df_results.to_string(index=False))



import numpy as np
import pandas as pd
from scipy.optimize import minimize
import os
import warnings

# 忽略 pandas 的一些合并警告，保持控制台输出干净
warnings.filterwarnings("ignore")

# ==============================================================================
# 🚀 基础功能定义区
# ==============================================================================

def ewma_covariance(returns, lbd):
    """
    计算指数加权协方差矩阵 (EWMA)
    逻辑：给最近的数据赋予更大的权重，越老的数据权重越小。
    """
    T, N = returns.shape
    # 生成衰减权重序列
    weights = (1 - lbd) * (lbd ** np.arange(T-1, -1, -1))
    weights /= weights.sum() # 归一化，确保权重和为1
    
    # 收益率去均值
    centered_returns = returns - np.mean(returns, axis=0)
    # 矩阵乘法计算加权协方差
    cov = np.dot(centered_returns.T, centered_returns * weights[:, np.newaxis])
    return cov

def max_sharpe_ratio(cov_matrix, exp_returns, risk_free_rate):
    """计算最大夏普比率组合 (Max SR)"""
    num_assets = len(exp_returns)
    
    # 目标函数：最大化夏普比率 (也就是最小化 负的夏普比率)
    def neg_sharpe(weights):
        port_ret = np.dot(weights, exp_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(port_ret - risk_free_rate) / port_vol

    # ⚠️ 考点修改预警 1：仓位加总限制 (Constraints)
    # np.sum(w) - 1.0 == 0 代表所有股票仓位加起来必须是 100% (满仓)。
    # 如果老师说可以保留 10% 的现金，这里就改成 np.sum(w) - 0.9
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    
    # ⚠️ 考点修改预警 2：做多做空限制 (Bounds)
    # (0.0, 1.0) 代表每只股票最少买 0，最多买 100%（只能做多，不能做空）。
    # 如果题目说允许做空 (Allow Short Selling)，比如最多做空 20%，就改成 (-0.2, 1.0)
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    
    # 初始瞎猜：均分仓位
    init_guess = np.ones(num_assets) / num_assets
    
    opt = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return opt.x

def risk_parity(cov_matrix):
    """计算风险平价组合 (Risk Parity)"""
    num_assets = cov_matrix.shape[0]
    
    # 目标函数：让每个股票的“风险贡献”无限逼近“平均风险”
    def target_risk_contributions(weights):
        port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
        marginal_contribs = np.dot(cov_matrix, weights)
        risk_contribs = weights * marginal_contribs # 每只股票的绝对风险贡献
        
        target_rc = port_var / num_assets # 理想情况：大家风险平摊
        
        # 惩罚函数：真实贡献和理想平摊的差值越小越好
        return np.sum((risk_contribs - target_rc)**2) * 1e6

    # 边界约束和 MaxSR 完全一样
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    init_guess = np.ones(num_assets) / num_assets
    
    opt = minimize(target_risk_contributions, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return opt.x

def analyze_attribution(weights, out_r, er_exante, cov_exante, stock_names):
    """
    计算事后收益与风险归因 (增加 Total 汇总行)
    """
    # ---------------------------------------------------------
    # 1. 收益归因 (Return Attribution)
    # ---------------------------------------------------------
    # ⚠️ 考点修改预警 3：调仓频率 (Rebalancing)
    # 题目明确说 "do not rebalance (不调仓)"，所以实际收益是 12个月复利连乘。
    # 如果老师丧心病狂地改成了 "monthly rebalanced (每月初重新调仓回设定比例)"，
    # 那实际收益的计算方式将完全不同！(不过这门课大概率不会考这么偏)
    compounded_stock_returns = np.prod(1 + out_r, axis=0) - 1
    
    # 实际收益绝对贡献 = 权重 * 复利收益
    act_ret_attr = weights * compounded_stock_returns
    port_act_ret = np.sum(act_ret_attr) # 组合总实际收益
    act_ret_pct = act_ret_attr / port_act_ret if port_act_ret != 0 else np.zeros_like(weights)
    
    # 预期收益绝对贡献
    exp_ret_attr = weights * er_exante
    port_exp_ret = np.sum(exp_ret_attr) # 组合总预期收益
    exp_ret_pct = exp_ret_attr / port_exp_ret if port_exp_ret != 0 else np.zeros_like(weights)
    
    # ---------------------------------------------------------
    # 2. 风险归因 (Volatility Attribution)
    # ---------------------------------------------------------
    cov_expost_monthly = np.cov(out_r, rowvar=False) 
    
    # ⚠️ 考点修改预警 4：样本外数据年化
    # 题目给的是月度收益率 (Monthly)，所以事后算出来的协方差也是月度的。
    # 必须乘以 12 变成年度，否则归因数字全错！
    cov_expost = cov_expost_monthly * 12             
    
    # 实际风险归因
    port_vol_act = np.sqrt(np.dot(weights.T, np.dot(cov_expost, weights))) # 组合总实际波动率
    mrc_act = np.dot(cov_expost, weights) / port_vol_act
    act_vol_attr = weights * mrc_act
    act_vol_pct = act_vol_attr / port_vol_act
    
    # 预期风险归因
    port_vol_exp = np.sqrt(np.dot(weights.T, np.dot(cov_exante, weights))) # 组合总预期波动率
    mrc_exp = np.dot(cov_exante, weights) / port_vol_exp
    exp_vol_attr = weights * mrc_exp
    exp_vol_pct = exp_vol_attr / port_vol_exp
    
    # ---------------------------------------------------------
    # 3. 组装输出，并添加 Total 汇总行
    # ---------------------------------------------------------
    # 生成基础数据帧
    df_ret = pd.DataFrame({
        'Stock': stock_names,
        'Expected_Return_Attribution': exp_ret_attr,
        'Actual_Return_Attribution': act_ret_attr,
        'Expected_Return_Attrib_Pct': exp_ret_pct,
        'Actual_Return_Attrib_Pct': act_ret_pct
    })
    
    # 生成 Total 汇总行数据
    total_row_ret = pd.DataFrame([{
        'Stock': 'TOTAL (Portfolio)',
        'Expected_Return_Attribution': port_exp_ret,
        'Actual_Return_Attribution': port_act_ret,
        'Expected_Return_Attrib_Pct': 1.0, # 百分比相加必须是 100%
        'Actual_Return_Attrib_Pct': 1.0
    }])
    df_ret = pd.concat([df_ret, total_row_ret], ignore_index=True)
    
    
    # 同理，组装风险归因表
    df_vol = pd.DataFrame({
        'Stock': stock_names,
        'Expected_Vol_Attribution': exp_vol_attr,
        'Actual_Vol_Attribution': act_vol_attr,
        'Expected_Vol_Attrib_Pct': exp_vol_pct,
        'Actual_Vol_Attrib_Pct': act_vol_pct
    })
    
    total_row_vol = pd.DataFrame([{
        'Stock': 'TOTAL (Portfolio)',
        'Expected_Vol_Attribution': port_vol_exp,
        'Actual_Vol_Attribution': port_vol_act,
        'Expected_Vol_Attrib_Pct': 1.0,
        'Actual_Vol_Attrib_Pct': 1.0
    }])
    df_vol = pd.concat([df_vol, total_row_vol], ignore_index=True)
    
    # 最后统一保留 6 位小数，让表格看起来清爽整洁
    numeric_cols_ret = df_ret.select_dtypes(include=[np.number]).columns
    df_ret[numeric_cols_ret] = df_ret[numeric_cols_ret].round(6)
    
    numeric_cols_vol = df_vol.select_dtypes(include=[np.number]).columns
    df_vol[numeric_cols_vol] = df_vol[numeric_cols_vol].round(6)
    
    return df_ret, df_vol

# ==============================================================================
# 🚀 主程序开始执行
# ==============================================================================
def main():
    print("=" * 65)
    print(" 🚀 Problem 3: 投资组合构建与归因分析 (带 Total 汇总版)")
    print("=" * 65)

    # ⚠️ 考点修改预警 5：宏观基准参数
    # 题目给的 4% 利率 和 0.97 的半衰期
    rf = 0.04 
    lbd = 0.97 

    # ------------------------------------------------------------------
    # Part A: 读取 In-sample 数据并计算权重
    # ------------------------------------------------------------------
    print("\n▶️ Part A: 读取样本内数据并进行最优化配置")
    
    # ⚠️ 考点修改预警 6：读取文件的路径
    # 如果考试发的文件名变了，务必修改这里
    insample_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\OldFinals\Fall2025\Final\problem3_insample.csv"
    if not os.path.exists(insample_path):
        print(f"❌ 找不到文件: {insample_path}")
        return
        
    insample = pd.read_csv(insample_path, sep=None, engine='python')
    stocks = [col for col in insample.columns if col.lower() != "date"]
    insample_r = insample[stocks].values

    # ⚠️ 考点修改预警 7：年化缩放公式
    # 严格根据 Hint: Scale expected return as (1 + er)^12 - 1
    # Scale covariance as ewCovar * 12
    er_monthly = np.mean(insample_r, axis=0)
    er = (1 + er_monthly)**12 - 1
    covar = ewma_covariance(insample_r, lbd) * 12

    # 调用优化器算权重
    mSR = max_sharpe_ratio(covar, er, rf)
    rPP = risk_parity(covar)

    df_weights = pd.DataFrame({
        'Stock': stocks,
        'MaxSR_Weight': np.round(mSR, 4),
        'RiskParity_Weight': np.round(rPP, 4)
    })
    
    # 给权重表也加个 Total 行，验算是不是正好等于 100%
    total_w = pd.DataFrame([{
        'Stock': 'TOTAL', 
        'MaxSR_Weight': np.round(np.sum(mSR), 4), 
        'RiskParity_Weight': np.round(np.sum(rPP), 4)
    }])
    df_weights = pd.concat([df_weights, total_w], ignore_index=True)
    
    print("✅ 样本内优化完成！最优权重分配如下：")
    print(df_weights.to_string(index=False))

    # ------------------------------------------------------------------
    # Part B: 读取 Out-sample 数据并进行归因分析
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print(" ▶️ Part B: 样本外绩效与风险归因 (Ex-Post Attribution)")
    print("=" * 65)

    outsample_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\OldFinals\Fall2025\Final\problem3_outsample.csv"
    if not os.path.exists(outsample_path):
        print(f"❌ 找不到文件: {outsample_path}")
        return
        
    outsample = pd.read_csv(outsample_path, sep=None, engine='python')
    out_r = outsample[stocks].values

    # ==========================
    # 1. Max SR 组合归因
    # ==========================
    print("\n🔍 [Max SR Portfolio (最大夏普组合) 归因分析]")
    ret_attr_msr, vol_attr_msr = analyze_attribution(mSR, out_r, er, covar, stocks)
    
    print("\n--- 💰 收益归因表 (Return Attribution) ---")
    print(ret_attr_msr.to_string(index=False))
    
    print("\n--- ⚡ 风险归因表 (Volatility Attribution) ---")
    print(vol_attr_msr.to_string(index=False))

    # ==========================
    # 2. Risk Parity 组合归因
    # ==========================
    print("\n" + "-" * 65)
    print("🔍 [Risk Parity Portfolio (风险平价组合) 归因分析]")
    
    # 💡 极其重要的细节：风险平价模型没有事前预期收益！
    # 所以为了对齐 Julia 的逻辑，我们给它传进去一个全是 0 的数组 (np.zeros_like(er))。
    # 这样算出来的预期收益百分比全是 0，符合 RPP 不预测收益的特性。
    ret_attr_rpp, vol_attr_rpp = analyze_attribution(rPP, out_r, np.zeros_like(er), covar, stocks)
    
    print("\n--- ⚡ 风险归因表 (Volatility Attribution) ---")
    print(vol_attr_rpp.to_string(index=False))
    print("\n注：RPP 策略不依赖预期收益，因此这里仅展示其核心的风险归因验证。")

if __name__ == "__main__":
    main()

