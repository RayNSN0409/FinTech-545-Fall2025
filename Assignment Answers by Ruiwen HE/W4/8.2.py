import pandas as pd
import numpy as np
from scipy import stats

# ------------------------------------------------------------------------------
# 1. Load Data
# ------------------------------------------------------------------------------
file_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test7_2.csv"
df = pd.read_csv(file_path)

# ------------------------------------------------------------------------------
# 2. Fit T-Distribution & Calculate VaR
# ------------------------------------------------------------------------------
# Fit T-distribution for each column (asset)
t_params = df.apply(lambda x: pd.Series(stats.t.fit(x.dropna()), index=['Nu', 'Mu', 'Scale'])).T

# Calculate 95% Quantile
# Quantile = Mu + t_ppf(0.05, Nu) * Scale
# We use .values for Nu to ensure correct broadcasting with stats.t.ppf
t_score = stats.t.ppf(0.05, df=t_params['Nu'].values)
var_quantile = t_params['Mu'] + t_score * t_params['Scale']

# Calculate Metrics
# VaR Absolute: |Quantile|
var_absolute = abs(var_quantile)

# VaR Diff from Mean: Mean - Quantile (Distance to tail)
var_diff_from_mean = t_params['Mu'] - var_quantile

# ------------------------------------------------------------------------------
# 3. Output
# ------------------------------------------------------------------------------
output_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W4\testout_8.2.csv"
output_df = pd.DataFrame({
    'VaR Absolute': var_absolute,
    'VaR Diff from Mean': var_diff_from_mean
})
output_df.to_csv(output_path, index=False)
print(output_df.to_string(index=False))