import pandas as pd
import numpy as np

# ------------------------------------------------------------------------------
# 1. Load Data
# ------------------------------------------------------------------------------
# Update path as needed
file_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test5_3.csv"
# file_path = "test5_3.csv" # Uncomment for testing in current dir
df = pd.read_csv(file_path)
input_cov = df.values

# ------------------------------------------------------------------------------
# 2. Higham (2002) Near-PSD Repair Functions
# ------------------------------------------------------------------------------

def project_to_positive_semidefinite(matrix):
    """
    Project onto Positive Semi-Definite (PSD) Cone.
    Logic: Eigen Decomposition -> Clip negative eigenvalues -> Reconstruct
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # Core: Clip negative eigenvalues to 0
    eigenvalues = np.maximum(eigenvalues, 0)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

def project_to_unit_diagonal(matrix):
    """
    Project onto the set of Unit Diagonal Matrices.
    Logic: Force diagonal elements to 1.0 (valid for correlation matrices).
    """
    matrix_out = matrix.copy()
    np.fill_diagonal(matrix_out, 1.0)
    return matrix_out

def compute_nearest_correlation_higham(target_correlation, tol=1e-9, max_iter=1000):
    """
    Higham (2002) Algorithm: Compute the nearest correlation matrix
    using Dykstra's Alternating Projections method.
    """
    # Initialize Dykstra's correction term (difference matrix)
    correction_matrix = np.zeros_like(target_correlation)
    
    # Current best estimate (initialize with input)
    current_correlation = target_correlation.copy()
    
    for k in range(max_iter):
        last_correlation = current_correlation.copy()
        
        # 1. Apply previous correction (R_k = Y_{k-1} - Delta S_{k-1})
        temp_matrix = last_correlation - correction_matrix
        
        # 2. Project onto PSD set (X_k = P_S(R_k))
        psd_projection = project_to_positive_semidefinite(temp_matrix)
        
        # 3. Update correction term (Delta S_k = X_k - R_k)
        correction_matrix = psd_projection - temp_matrix
        
        # 4. Project onto Unit Diagonal set (Y_k = P_U(X_k))
        current_correlation = project_to_unit_diagonal(psd_projection)
        
        # 5. Check Convergence (Frobenius Norm)
        diff = np.linalg.norm(current_correlation - last_correlation, 'fro')
        if diff < tol:
            break
            
    return current_correlation

def fix_non_psd_covariance_higham(cov_matrix):
    """
    Main Wrapper: Fix a non-PSD covariance matrix using Higham's method.
    Workflow: Cov -> Vol + Corr -> Higham Fix -> New Cov
    """
    # 1. Extract Standard Deviations (Volatilities)
    std_devs = np.sqrt(np.diag(cov_matrix))
    # Avoid division by zero if any volatility is 0
    std_devs[std_devs < 1e-8] = 1e-8
    
    # 2. Convert to Correlation Matrix
    # Construct denominator matrix (std_i * std_j)
    vol_product_matrix = np.outer(std_devs, std_devs)
    raw_correlation = cov_matrix / vol_product_matrix

    # 3. Fix Correlation Matrix using Higham's Algorithm (Core Step)
    fixed_correlation = compute_nearest_correlation_higham(raw_correlation)
    
    # 4. Restore Covariance Matrix
    # Fixed Cov = Fixed Corr * (std_i * std_j)
    fixed_covariance = fixed_correlation * vol_product_matrix
    
    return fixed_covariance

# Apply the Higham fix
psd_cov_matrix = fix_non_psd_covariance_higham(input_cov)

# Convert back to DataFrame for viewing
psd_cov_df = pd.DataFrame(psd_cov_matrix, index=df.columns, columns=df.columns)
print("=== Higham Fixed Near-PSD Covariance Matrix ===")
print(psd_cov_df)

# ------------------------------------------------------------------------------
# 3. Simulation
# ------------------------------------------------------------------------------
n_assets = psd_cov_matrix.shape[0]
n_simulations = 100000

# Cholesky Decomposition (Now successful on the fixed PSD matrix)
try:
    L = np.linalg.cholesky(psd_cov_matrix)
except np.linalg.LinAlgError:
    # Fallback for extreme edge cases (numerical precision issues)
    print("Warning: Matrix still numerically unstable. Adding jitter.")
    psd_cov_matrix += np.eye(n_assets) * 1e-9
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

output_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W4\testout_5.4.csv"
output_df.to_csv(output_path, index=False)
print(f"File saved to: {output_path}")

# ==============================================================================
# Analysis of Differences
# ==============================================================================
#
# 1. Comparison Hierarchy:
#    Input (Non-PSD)  !=  Fixed_PSD_Matrix (Higham)  ~=  Output (Simulated)
#
# 2. Input vs. Fixed Matrix:
#    - The original input was mathematically invalid (Non-PSD).
#    - We applied Higham's (2002) algorithm, which finds the "nearest" correlation
#      matrix in terms of the Frobenius norm. This is mathematically more rigorous
#      than simple spectral clipping (Rebonato & Jackel) for finding the optimal projection.
#
# 3. Fixed Matrix vs. Output:
#    - The simulated output is statistically consistent with the Higham-fixed matrix.
#    - Small discrepancies are due to Monte Carlo sampling error (N=100,000).