import numpy as np
import pandas as pd

returns_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test11_1_returns.csv"
weights_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test11_1_weights.csv"
output_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W13\testout11_1.csv"

df_returns = pd.read_csv(returns_path)
df_weights = pd.read_csv(weights_path)

returns = df_returns.values
weights = df_weights.values.flatten()
assets = df_returns.columns.tolist()

n_periods, n_assets = returns.shape

w_t = np.zeros((n_periods, n_assets))
c_t = np.zeros((n_periods, n_assets))
R_p = np.zeros(n_periods)

current_value = weights.copy()

for t in range(n_periods):
    total_val = np.sum(current_value)
    w_start = current_value / total_val
    w_t[t] = w_start
    
    r_t = returns[t]
    
    daily_contrib = w_start * r_t
    c_t[t] = daily_contrib
    R_p[t] = np.sum(daily_contrib)
    
    current_value = current_value * (1 + r_t)

asset_total_returns = np.prod(1 + returns, axis=0) - 1
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

return_attrib = np.sum(c_t * (k_t / K)[:, None], axis=0)

std_Rp = np.std(R_p, ddof=1)

vol_attrib = np.zeros(n_assets)
for i in range(n_assets):
    cov_val = np.cov(c_t[:, i], R_p, ddof=1)[0, 1]
    vol_attrib[i] = cov_val / std_Rp

out_data = {
    'Value': ['TotalReturn', 'Return Attribution', 'Vol Attribution']
}

for i, asset in enumerate(assets):
    out_data[asset] = [asset_total_returns[i], return_attrib[i], vol_attrib[i]]

out_data['Portfolio'] = [port_total_return, port_total_return, std_Rp]

out_df = pd.DataFrame(out_data)
out_df.to_csv(output_path, index=False, float_format='%.9f')

print(out_df.to_string(index=False))