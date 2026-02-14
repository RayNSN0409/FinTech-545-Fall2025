# --- 1. 基础数据处理与矩阵运算 ---
import numpy as np                 # 核心数学库：矩阵运算、向量化计算、随机数生成（蒙特卡洛核心）
import pandas as pd                # 时间序列处理：DataFrame操作、日期索引管理、移动窗口计算

# --- 2. 统计与概率分布 ---
import scipy.stats as stats        # 统计分布库：拟合分布（t分布, 正态等）、假设检验（KS test, JB test）
from scipy.optimize import minimize # 数值优化：用于最大似然估计 (MLE) 和求解非线性方程

# --- 3. 计量经济学与风险建模 ---
import statsmodels.api as sm       # 计量工具：回归分析 (OLS)、ACF/PACF 图、QQ图
from statsmodels.tsa.arima.model import ARIMA # 时间序列均值方程建模
from arch import arch_model        # 波动率建模核心库：GARCH, EGARCH, TARCH 等异方差模型

# --- 4. 可视化 (学术风格) ---
import matplotlib.pyplot as plt    # 基础绘图
import seaborn as sns              # 高级统计绘图：热力图、分布图

# 设置绘图风格 (推荐)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6) # 设置默认图片大小

# --- 5. 数据读取函数示例 ---
def load_pure_data(path):
    """
    读取纯数值数据。
    默认假设：文件没有表头 (header=None)。
    """  
    # 第一行是表头 (如 AssetA, AssetB...)
    # 如果文件有表头，取消下面这行的注释，注释掉上面那行
    df = pd.read_csv(path, header=0)

    print(f"纯数据加载: {df.shape}")
    return df

import pandas as pd

def load_time_series(path):
    """
    读取时间序列数据。
    默认假设：第一行是表头 (header=0)，第一列是时间。
    """
    # --- 1. 读取数据 ---
    
    # [默认情况] 假设第一行是表头 (如: Date, SPX, VIX...)
    df = pd.read_csv(path, header=0)
    
    # [备选情况] 如果文件没有表头 (第一行就是数据)，请取消下方注释并注释掉上面一行
    # df = pd.read_csv(path, header=None)
    # # 如果没有表头，需要手动把第一列命名为 Date，防止后续报错
    # df.rename(columns={0: 'Date'}, inplace=True) 

    # --- 2. 规范化列名 ---
    # 无论原文件第一列叫 "Time", "Day", 还是 "Date"，统一重命名为 "Date"
    # 这样做是为了保证后续代码通用性
    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

    # --- 3. 时间索引处理 (QRM 核心步骤) ---
    # 必须转为 datetime 对象，否则 rolling() 等函数无法识别时间窗口
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 设为索引并排序
    # QRM 注释: 时间乱序会导致收益率计算错误 (Rt = Pt - Pt-1)，必须 sort_index
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    print(f"数据加载完成: {df.shape} | 时间范围: {df.index.min().date()} 至 {df.index.max().date()}")
    return df

# --- 测试调用 ---
file_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test7_1.csv"
df_prices = load_time_series(file_path)

# 计算对数收益率
import numpy as np
import pandas as pd

def calculate_log_returns_robust(df):
    """
    【严谨版】计算对数收益率 (Log Returns)。
    增加了对 0 和 负数价格 的防御性处理。
    """
    # --- 1. 数据清洗 (Sanity Check) ---
    # QRM 铁律: 资产价格必须严格为正 (Price > 0) 才能做对数运算
    # 任何 <= 0 的价格通常意味着: 
    #   a) 数据错误 (Data Glitch)
    #   b) 资产退市/破产 (Bankruptcy)
    #   c) 极端行情 (如2020年原油负油价，但在常规模型中需作为异常值处理)
    
    # 将所有 <= 0 的价格强制标记为 NaN (空值)
    # 这样它们就不会参与运算，也不会生成 inf
    clean_df = df.where(df > 0, np.nan) 
    
    # 检查是否有数据被剔除 (可选，用于调试)
    if df.shape != clean_df.dropna().shape:
        # 实际代码中可以用 logging 记录，这里为了简洁省略
        pass

    # --- 2. 核心计算 ---
    # 由于我们把非法值变为了 NaN，np.log(NaN) 结果还是 NaN，不会报错
    log_ret = np.log(clean_df / clean_df.shift(1))
    
    # --- 3. 最终清洗 ---
    # 移除第一行 (必然是NaN) 和中间产生的任何 NaN (由价格<=0导致)
    return log_ret.dropna()

def calculate_arithmetic_returns_robust(df):
    """
    【严谨版】计算算术收益率。
    处理分母为0的情况，并清洗非法价格。
    """
    # 虽然算术收益率允许价格为负，但在风险模型中，
    # 负价格会导致“收益率”概念失效 (基数变为负数)，建议同样清洗。
    clean_df = df.where(df > 0, np.nan)
    
    prev_prices = clean_df.shift(1)
    
    # 计算
    arith_ret = (clean_df - prev_prices) / prev_prices
    
    # 双重保险: 将计算产生的 inf (如果漏网之鱼) 替换为 NaN
    arith_ret.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return arith_ret.dropna()

# 描述性分析函数
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_risk_stats(returns_df):
    """
    计算 QRM 核心描述性统计量。
    输入: 收益率 DataFrame (Clean Log Returns)
    输出: 统计指标汇总表
    """
    # 初始化结果表
    summary = pd.DataFrame(index=returns_df.columns)
    
    # 1. 基础指标 (均值、中位数、标准差)
    summary['Mean'] = returns_df.mean()
    summary['Median'] = returns_df.median()
    summary['Std_Dev'] = returns_df.std()  # 日波动率
    
    # 2. 高阶矩 (风险建模的关键)
    # Skewness (偏度): 
    #   < 0 (负偏): 左尾更长 -> 发生暴跌的概率比暴涨大 (股票典型特征)
    #   > 0 (正偏): 右尾更长 -> 就像买彩票
    summary['Skewness'] = returns_df.skew()
    
    # Kurtosis (峰度 - Fisher定义): 
    #   Pandas 默认计算 "超额峰度" (Excess Kurtosis)
    #   Normal = 0. 如果 > 0，说明是"尖峰肥尾" (Fat Tails) -> 极端风险高
    summary['Excess_Kurtosis'] = returns_df.kurtosis()
    
    # 3. 正态性检验 (Jarque-Bera Test)
    # 假设检验 H0: 数据服从正态分布
    # 如果 P-value < 0.05，拒绝 H0 -> 数据不是正态的 (必须用历史模拟法或 t分布模型)
    
    # 使用 apply 对每一列进行 JB 测试
    # stats.jarque_bera 返回 (statistic, p-value)
    jb_results = returns_df.apply(lambda x: stats.jarque_bera(x)[1])
    summary['JB_P_Value'] = jb_results
    
    # 增加一个直观的判断列
    summary['Is_Normal'] = summary['JB_P_Value'] > 0.05
    
    return summary

# 若JB_P_Value < 0.05，则拒绝正态性假设，构建T分布模型
import numpy as np
import pandas as pd
import scipy.stats as stats

def fit_t_distribution_metrics(data, confidence_level=0.99):
    """
    拟合 T 分布并提取 QRM 关键统计量。
    
    输入: 
        data: 收益率序列或残差 (Series/Array)
        confidence_level: VaR 置信度 (如 0.99)
        
    输出:
        Series 包含: 拟合参数(DoF, Loc, Scale) + 理论统计量(Var, Kurt) + 风险指标(VaR)
    """
    # --- 1. 拟合 (Fitting) ---
    # 使用 MLE (最大似然) 估计三个参数
    # nu (df): 自由度 -> 决定尾部厚度。nu越小，尾部越肥 (越危险)。
    # mu (loc): 位置参数 -> 类似于均值。
    # sigma (scale): 尺度参数 -> 类似于标准差，但受 nu 影响。
    nu, mu, sigma = stats.t.fit(data)
    
    # --- 2. 提取理论统计量 (Derived Statistics) ---
    # T分布的方差 != scale^2。公式: Var = scale^2 * (nu / (nu-2))
    # 只有当 nu > 2 时方差才存在，否则为无穷大 (极其危险的市场)
    implied_var = sigma**2 * (nu / (nu - 2)) if nu > 2 else np.inf
    
    # 理论超额峰度 (Excess Kurtosis)。公式: 6 / (nu - 4)
    # 只有当 nu > 4 时存在。注意: 正态分布 nu -> inf, 峰度 -> 0
    implied_kurt = 6 / (nu - 4) if nu > 4 else np.inf
    
    # --- 3. 计算风险指标 (VaR) ---
    # 这步是拟合 T 分布的最终目的: 用它来算 VaR
    # ppf: 百分位点函数 (CDF的逆函数)。计算左尾 alpha 处的损失。
    alpha = 1 - confidence_level
    t_var = stats.t.ppf(alpha, df=nu, loc=mu, scale=sigma)
    
    # --- 4. 整理结果 ---
    metrics = pd.Series({
        "DoF (nu)": nu,                # 核心: <5 表示极度肥尾; >30 近似正态
        "Location (mu)": mu,
        "Scale (sigma)": sigma,        # 注意: 这不是标准差
        "Implied_Vol": np.sqrt(implied_var), # 理论波动率
        "Implied_Kurt": implied_kurt,  # 理论峰度
        f"VaR_{int(confidence_level*100)}%": t_var # 基于T分布的风险值
    })
    
    return metrics
    
# OLS建模、残差提取与T分布拟合示例
import statsmodels.api as sm
import scipy.stats as stats
import pandas as pd
import numpy as np

def run_ols_and_jb_test(df, y_col, x_cols):
    """
    执行 OLS 回归并自动对残差进行 Jarque-Bera 测试。
    
    参数:
        df: 包含数据的 DataFrame
        y_col: 因变量列名 (字符串)
        x_cols: 自变量列名列表 (字符串列表)
        
    返回:
        model: 训练好的 OLS 模型对象
        jb_p_value: 残差的 JB 测试 P值
    """
    # 1. 准备数据
    Y = df[y_col]
    X = df[x_cols]
    
    # [关键] Statsmodels 默认不加截距项(Intercept)，必须手动添加！
    X = sm.add_constant(X)
    
    # 2. 拟合 OLS 模型
    model = sm.OLS(Y, X).fit()
    
    # 3. 提取残差 (Residuals)
    # 残差 = 真实值 Y - 预测值 Y_hat
    residuals = model.resid
    
    # 4. 执行 JB Test
    # H0: 残差服从正态分布
    # H1: 残差不服从正态分布 (通常是肥尾)
    jb_stat, jb_p_value = stats.jarque_bera(residuals)
    
    # --- 打印报告 ---
    print("="*40)
    print(f"OLS Regression Analysis (Y={y_col})")
    print("="*40)
    print(f"R-squared: {model.rsquared:.4f}")
    print(f"Params:\n{model.params}")
    print("\n--- Residual Diagnostics (JB Test) ---")
    print(f"JB Statistic: {jb_stat:.4f}")
    print(f"JB P-Value:   {jb_p_value:.4e}")
    
    if jb_p_value < 0.05:
        print(">> 结果: ❌ 拒绝正态假设 (P < 0.05)")
        print(">> 含义: 残差存在【肥尾】或【偏度】。OLS 标准误可能失效。CLT 只能救 Beta 的分布，救不了残差的尾部厚度。尤其在Risk领域，风控师关心残差的尾巴，用 OLS 算风险，会严重低估尾部风险，而MLE 算出的自由度能精确捕捉这个尾部风险。")
        print(">> 建议: 转用 MLE T-Regression (Robust Regression)。")
    else:
        print(">> 结果: ✅ 无法拒绝正态假设 (P >= 0.05)")
        print(">> 含义: 残差符合正态分布，OLS 模型有效。")
        
    print("="*40)
    
    return model, jb_p_value 

# MLE T-回归示例
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize

def fit_t_regression_mle(df, y_col, x_cols):
    """
    执行基于 Student's t 分布误差假设的 MLE 线性回归。
    
    适用场景: 当 OLS 残差未通过 JB Test (肥尾/异常值) 时。
    
    参数:
        df: 数据表
        y_col: 因变量列名
        x_cols: 自变量列名列表
        
    返回:
        results_df: 包含 Beta系数, Nu (自由度), Sigma (尺度) 的结果表
        model_metrics: 包含 AIC, BIC, Log-Likelihood 的字典
    """
    # 1. 数据准备
    Y = df[y_col].values
    X = df[x_cols]
    X = sm.add_constant(X) # 必须加常数项
    X_mat = X.values
    
    num_vars = X_mat.shape[1] # 变量数量 (含截距)
    
    # 2. 定义负对数似然函数 (Objective Function)
    # 我们要最小化这个函数 -> 等同于最大化似然概率
    def nll_t_dist(params):
        # 参数拆解: 前面的 params 是回归系数 Beta，最后两个是 Nu 和 Sigma
        betas = params[:num_vars]
        nu = params[num_vars]
        sigma = params[num_vars+1]
        
        # 物理约束: 自由度和尺度必须 > 0
        if nu <= 0 or sigma <= 0:
            return np.inf
        
        # 计算残差: Y - X*Beta
        residuals = Y - (X_mat @ betas)
        
        # 计算对数似然 (Log Likelihood)
        # 使用 scipy 的 t.logpdf 计算在当前 nu, sigma 下残差出现的概率
        ll = stats.t.logpdf(residuals, df=nu, loc=0, scale=sigma)
        
        # 返回负数 (因为 minimize 是求最小值)
        return -np.sum(ll)

    # 3. 聪明地选择起点 (Warm Start)
    # 先跑一个 OLS，用它的 Beta 作为起点，避免优化器迷路
    print(">> 正在运行 OLS 获取初始猜测值...")
    ols_model = sm.OLS(Y, X).fit()
    beta_init = ols_model.params.values
    
    # 对 OLS 残差拟合 T分布，获取 Nu 和 Sigma 的初始值
    nu_init, _, sigma_init = stats.t.fit(ols_model.resid)
    
    # 拼接所有初始参数
    initial_guess = np.append(beta_init, [nu_init, sigma_init])
    
    # 4. 设置边界 (Bounds)
    # Beta: 无限制 (None, None)
    # Nu, Sigma: 必须 > 0.0001 (避免除以0错误)
    bounds = [(None, None)] * num_vars + [(0.001, None), (0.001, None)]
    
    # 5. 开始优化 (Run Optimization)
    print(f">> 开始 MLE 优化 (样本量: {len(Y)})...")
    result = minimize(nll_t_dist, initial_guess, method='L-BFGS-B', bounds=bounds)
    
    if not result.success:
        print(f"!! 警告: 优化失败 -> {result.message}")
        return None
    
    # 6. 整理结果
    final_params = result.x
    
    # 提取回归系数
    betas_result = pd.Series(final_params[:num_vars], index=X.columns)
    
    # 提取分布参数
    nu_est = final_params[num_vars]
    sigma_est = final_params[num_vars+1]
    
    # 汇总输出
    print("\n" + "="*40)
    print("MLE T-Regression Results (Robust)")
    print("="*40)
    print(f"Degrees of Freedom (Nu): {nu_est:.4f}")
    print(f"Scale (Sigma):           {sigma_est:.4f}")
    print("-" * 40)
    print("Coefficients (Betas):")
    print(betas_result)
    print("="*40)
    
    # 构造返回对象
    summary_series = betas_result.copy()
    summary_series['Nu (df)'] = nu_est
    summary_series['Sigma'] = sigma_est
    
    return summary_series
