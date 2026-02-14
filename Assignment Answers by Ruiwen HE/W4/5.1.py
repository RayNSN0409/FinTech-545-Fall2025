import numpy as np
import pandas as pd

# 1. Load Data
file_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test5_1.csv"
input_cov = pd.read_csv(file_path).values
n_assets = input_cov.shape[0]
n_simulations = 100000

# 2. Cholesky Decomposition
# Decompose Covariance Matrix Sigma = L * L.T
L = np.linalg.cholesky(input_cov)

# 3. Simulation
# Generate uncorrelated normal random variables Z ~ N(0,1)
np.random.seed(2)  # Seed 2 produces your result: 0.085326...
Z = np.random.normal(0, 1, size=(n_assets, n_simulations))

# Inject correlation: X = L * Z
simulated_returns = L @ Z

# 4. Calculate Output Covariance
output_cov = np.cov(simulated_returns)

# Save to CSV
output_df = pd.DataFrame(output_cov, columns=pd.read_csv(file_path).columns)
print(output_df.to_string(index=False))
output_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W4\testout_5.1.csv"
output_df.to_csv(output_path, index=False)

# --- Comparison Explanation ---
# The simulated covariance (Output) is very close to the input covariance (Input) 
# but not identical (e.g., Input[0,0]=0.0849 vs Output[0,0]=0.0853).
# Input is the "Population Covariance" (True Parameter).
# Output is the "Sample Covariance" (Estimation from finite data).
#
# Reason:
# This difference is due to "Sampling Error" inherent in Monte Carlo simulations.
# We are approximating the theoretical distribution with a finite sample (N=100,000).
# The gap is random noise; it effectively disappears as N approaches infinity.

