import numpy as np
import pandas as pd

beta_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test11_2_beta.csv"
factor_returns_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test11_2_factor_returns.csv"
stock_returns_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test11_2_stock_returns.csv"
weights_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test11_2_weights.csv"
output_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W13\testout_11.2.csv"

df_stocks = pd.read_csv(stock_returns_path)
df_factors = pd.read_csv(factor_returns_path)
df_betas = pd.read_csv(beta_path, index_col='Stock')
df_weights = pd.read_csv(weights_path)

R_s = df_stocks.values
R_f = df_factors.values
B = df_betas.values
weights = df_weights.values.flatten()

n_periods, n_stocks = R_s.shape
n_factors = R_f.shape[1]

w_t = np.zeros((n_periods, n_stocks))
c_t_s = np.zeros((n_periods, n_stocks))
R_p = np.zeros(n_periods)

current_value = weights.copy()

for t in range(n_periods):
    w_start = current_value / np.sum(current_value)
    w_t[t] = w_start
    r_t = R_s[t]
    c_t_s[t] = w_start * r_t
    R_p[t] = np.sum(c_t_s[t])
    current_value = current_value * (1 + r_t)

port_total_return = np.sum(current_value) / np.sum(weights) - 1

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

factor_total_returns = np.prod(1 + R_f, axis=0) - 1

c_t_f = np.zeros((n_periods, n_factors))
c_t_alpha = np.zeros(n_periods)

for t in range(n_periods):
    for j in range(n_factors):
        c_t_f[t, j] = np.sum(w_t[t] * B[:, j] * R_f[t, j])
    c_t_alpha[t] = R_p[t] - np.sum(c_t_f[t])

return_attrib_f = np.sum(c_t_f * (k_t / K)[:, None], axis=0)
return_attrib_alpha = np.sum(c_t_alpha * (k_t / K))

tr_alpha = np.prod(1 + c_t_alpha) - 1

std_Rp = np.std(R_p, ddof=1)

vol_attrib_f = np.zeros(n_factors)
for j in range(n_factors):
    cov_val = np.cov(c_t_f[:, j], R_p, ddof=1)[0, 1]
    vol_attrib_f[j] = cov_val / std_Rp

cov_val_alpha = np.cov(c_t_alpha, R_p, ddof=1)[0, 1]
vol_attrib_alpha = cov_val_alpha / std_Rp

out_data = {
    'Value': ['TotalReturn', 'Return Attribution', 'Vol Attribution'],
    'F1': [factor_total_returns[0], return_attrib_f[0], vol_attrib_f[0]],
    'F2': [factor_total_returns[1], return_attrib_f[1], vol_attrib_f[1]],
    'F3': [factor_total_returns[2], return_attrib_f[2], vol_attrib_f[2]],
    'Alpha': [tr_alpha, return_attrib_alpha, vol_attrib_alpha],
    'Portfolio': [port_total_return, port_total_return, std_Rp]
}

out_df = pd.DataFrame(out_data)
out_df.to_csv(output_path, index=False, float_format='%.9f')

print(out_df.to_string(index=False))