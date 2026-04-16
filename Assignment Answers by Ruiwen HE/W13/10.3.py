import numpy as np
import pandas as pd
from scipy.optimize import minimize

input_cov_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test5_2.csv"
input_mean_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test10_3_means.csv"
output_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W13\testout_10.3.csv"

def negative_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate):
    port_return = weights.T @ expected_returns
    port_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
    return -(port_return - risk_free_rate) / port_volatility

def get_max_sharpe_weights(expected_returns, cov_matrix, risk_free_rate):
    n = cov_matrix.shape[0]
    init_guess = np.ones(n) / n
    bounds = tuple((0.0, 1.0) for _ in range(n))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    
    result = minimize(
        negative_sharpe_ratio,
        init_guess,
        args=(expected_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-12, 'disp': False}
    )
    return result.x

df_cov = pd.read_csv(input_cov_path)
cov_matrix = df_cov.values

df_mean = pd.read_csv(input_mean_path)
expected_returns = df_mean.values.flatten()

risk_free_rate = 0.04

optimal_weights = get_max_sharpe_weights(expected_returns, cov_matrix, risk_free_rate)

df_out = pd.DataFrame(optimal_weights, columns=['W'])
df_out.to_csv(output_path, index=False)

print("W")
for w in optimal_weights:
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