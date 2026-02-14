import pandas as pd
import numpy as np

# ------------------------------------------------------------------------------
# 1. Load Data
# ------------------------------------------------------------------------------
file_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test5_3.csv"
df = pd.read_csv(file_path)
input_cov = df.values

# ------------------------------------------------------------------------------
# 2. Near-PSD Repair Function (Rebonato & Jäckel)
# ------------------------------------------------------------------------------
def get_near_psd(covariance_matrix):
    """
    Adjusts a non-PSD covariance matrix to be Positive Semi-Definite (PSD)
    using the Rebonato & Jäckel spectral decomposition method.
    """
    # 1. Extract Standard Deviations (Volatilities)
    std_devs = np.sqrt(np.diag(covariance_matrix))
    
    # 2. Convert to Correlation Matrix
    # Normalize: Corr = D^-1 * Cov * D^-1
    # Using outer product for efficient broadcasting
    corr_matrix = covariance_matrix / np.outer(std_devs, std_devs)
    
    # 3. Spectral Decomposition
    # Use eigh for symmetric matrices (more stable than eig)
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    
    # 4. Fix Eigenvalues
    # Force negative eigenvalues to zero (or a small epsilon)
    eigenvalues = np.maximum(eigenvalues, 0)
    
    # 5. Reconstruct Correlation Matrix
    # T = Q * Lambda * Q^T
    # The resulting matrix T is PSD, but diagonal elements may not be exactly 1
    raw_corr_psd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    # 6. Rescale Diagonals (The "Rebonato & Jäckel" Step)
    # Force diagonal elements back to 1.0 to ensure a valid correlation matrix
    # S = diag(1 / sqrt(diag(T)))
    # Final Corr = S * T * S
    d = np.sqrt(np.diag(raw_corr_psd))
    final_corr_psd = raw_corr_psd / np.outer(d, d)
    
    # 7. Restore Covariance Matrix
    # Cov = D * Final_Corr * D
    final_cov_psd = np.outer(std_devs, std_devs) * final_corr_psd
    
    return final_cov_psd

# Apply the fix
psd_cov_matrix = get_near_psd(input_cov)

# View the fixed matrix
psd_cov_df = pd.DataFrame(psd_cov_matrix, index=df.columns, columns=df.columns)
print("=== Near PSD Covariance Matrix (Fixed) ===")
print(psd_cov_df)

# ------------------------------------------------------------------------------
# 3. Simulation
# ------------------------------------------------------------------------------
n_assets = psd_cov_matrix.shape[0]
n_simulations = 100000

# Cholesky Decomposition (Now successful on the fixed PSD matrix)
L = np.linalg.cholesky(psd_cov_matrix)

# Generate Random Numbers
np.random.seed(2)  # Fixed seed for reproducibility
Z = np.random.normal(0, 1, size=(n_assets, n_simulations))

# Inject Correlation: X = L * Z
simulated_returns = L @ Z

# ------------------------------------------------------------------------------
# 4. Calculate Output & Save
# ------------------------------------------------------------------------------
output_cov = np.cov(simulated_returns)
output_df = pd.DataFrame(output_cov, columns=df.columns)

print("\n=== Simulated Output Covariance ===")
print(output_df)

output_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W4\testout_5.3.csv"
output_df.to_csv(output_path, index=False)
print(f"File saved to: {output_path}")

# ==============================================================================
# Analysis of Differences
# ==============================================================================
#
# 1. Comparison Hierarchy:
#    Input (Non-PSD)  !=  Fixed_PSD_Matrix  ~=  Output (Simulated)
#
# 2. Input vs. Fixed Matrix:
#    - The original input was mathematically invalid (Non-PSD), likely containing
#      conflicting correlations (e.g., violations of the triangle inequality).
#    - The 'psd_cov_matrix' is the close mathematically valid approximation.
#    - Differences here are necessary corrections, not errors.
#
# 3. Fixed Matrix vs. Output:
#    - The simulated output matches the 'psd_cov_matrix' very closely.
#    - Any remaining small differences are due to "Sampling Error" from the
#      finite number of simulations (N=100,000). These would vanish as N -> infinity.