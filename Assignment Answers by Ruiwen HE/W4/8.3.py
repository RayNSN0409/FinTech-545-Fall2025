import pandas as pd
import numpy as np
from scipy import stats

# ------------------------------------------------------------------------------
# 1. Load Data
# ------------------------------------------------------------------------------
file_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test7_2.csv"
# file_path = "test7_2.csv" # Uncomment for testing
df = pd.read_csv(file_path)

# ------------------------------------------------------------------------------
# 2. Fit T-Distribution & Calculate VaR (Monte Carlo)
# ------------------------------------------------------------------------------
# Fit T-distribution for each column
# Returns DataFrame: Index=Assets, Columns=[Nu, Mu, Scale]
t_params = df.apply(lambda x: pd.Series(stats.t.fit(x.dropna()), index=['Nu', 'Mu', 'Scale'])).T

# Monte Carlo Simulation Parameters
n_simulations = 100000
np.random.seed(2) 

# Generate Random Variables (T-Distributed)
# Broadcasting works here: Nu values (N,) broadcast against size (M, N)
# Z shape will be (n_simulations, n_assets)
# Corrected: Use t_params['Nu'] instead of .loc['Nu']
Z = np.random.standard_t(t_params['Nu'].values, size=(n_simulations, len(df.columns)))

# Calculate Simulated Returns
# Formula: R = Mu + Z * Scale
simulated_returns = t_params['Mu'].values + Z * t_params['Scale'].values

# Calculate 5% Quantile (VaR threshold)
# axis=0 means calculate across the 100,000 simulations for each asset
VaR_95_quantile = np.percentile(simulated_returns, 5, axis=0)

# Calculate Metrics
VaR_absolute = np.abs(VaR_95_quantile)
VaR_diff_from_mean = t_params['Mu'].values - VaR_95_quantile

# ------------------------------------------------------------------------------
# 3. Output
# ------------------------------------------------------------------------------
output_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W4\testout_8.3.csv"

output_df = pd.DataFrame({
    'VaR Absolute': VaR_absolute,
    'VaR Diff from Mean': VaR_diff_from_mean
}, index=df.columns)

output_df.to_csv(output_path, index=False, header=True)

print(output_df.to_string(index=False))

# ==============================================================================
# 4. Comparison Analysis (Simulation vs Parametric)
# ==============================================================================
#
# *** Results Comparison ***
# 1. Task 8.2 (Parametric Calculation):
#    - Method: Exact mathematical formula using T-distribution CDF/PPF.
#    - Result: VaR Absolute = 0.041530 | VaR Diff from Mean = 0.087470
#
# 2. Task 8.3 (Monte Carlo Simulation - Current):
#    - Method: Generating 100,000 random samples based on fitted T-params.
#    - Result: VaR Absolute = 0.041476 | VaR Diff from Mean = 0.087416
#
# *** Analysis of Differences ***
# - Precision: The difference is approx 0.00005 (0.13%).
# - Reason: This slight discrepancy is due to "Sampling Error" (Monte Carlo noise).
#   We are approximating a continuous theoretical distribution with a finite 
#   discrete sample (N=100,000). 
# - Conclusion: The simulation results confirm the validity of the parametric model.
#   The simulated values have successfully converged to the theoretical values.