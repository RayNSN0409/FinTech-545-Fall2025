import numpy as np
import pandas as pd
from scipy.optimize import minimize

input_cov_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test5_2.csv"
input_mean_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test10_3_means.csv"
output_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W13\testout_10.4.csv"

def negative_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate):
    port_return = weights.T @ expected_returns
    port_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
    return -(port_return - risk_free_rate) / port_volatility

def get_max_sharpe_weights_bounded(expected_returns, cov_matrix, risk_free_rate):
    n = cov_matrix.shape[0]
    init_guess = np.ones(n) / n
    bounds = tuple((0.1, 0.5) for _ in range(n))
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

optimal_weights = get_max_sharpe_weights_bounded(expected_returns, cov_matrix, risk_free_rate)

df_out = pd.DataFrame(optimal_weights, columns=['W'])
df_out.to_csv(output_path, index=False, float_format='%.1f')

print("W")
for w in optimal_weights:
    print(f"{w:.1f}")