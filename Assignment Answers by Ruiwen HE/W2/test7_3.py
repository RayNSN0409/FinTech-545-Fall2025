import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import t
from scipy.optimize import minimize
import os

# Setup paths
input_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test7_3.csv"
output_dir = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W2"

# Load data
df = pd.read_csv(input_path)
Y = df['y'].values
X = sm.add_constant(df[['x1', 'x2', 'x3']]).values

# Define Negative Log-Likelihood (NLL) function
def nll(params):
    betas = params[:4]    # Alpha, B1, B2, B3
    nu = params[4]        # Degrees of Freedom
    sigma = params[5]     # Scale
    
    if nu <= 0 or sigma <= 0:
        return np.inf

    residuals = Y - (X @ betas)
    return -np.sum(t.logpdf(residuals, df=nu, loc=0, scale=sigma))

# Initial guess using OLS
ols_model = sm.OLS(Y, X).fit()
beta_init = ols_model.params
nu_init, _, sigma_init = t.fit(ols_model.resid)

initial_params = np.append(beta_init, [nu_init, sigma_init])
bounds = [(None, None)] * 4 + [(0.001, None), (0.001, None)]

# Run Optimization (MLE)
result = minimize(nll, initial_params, method='L-BFGS-B', bounds=bounds)

# Extract results
alpha, b1, b2, b3 = result.x[:4]
nu = result.x[4]
sigma = result.x[5]
mu = 0.0

# Print results
print(f"mu:    {mu:.6f}")
print(f"sigma: {sigma:.6f}")
print(f"nu:    {nu:.6f}")
print(f"Alpha: {alpha:.6f}")
print(f"B1:    {b1:.6f}")
print(f"B2:    {b2:.6f}")
print(f"B3:    {b3:.6f}")

# Save to Excel
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_file = os.path.join(output_dir, "t_regression_results_mle.xlsx")
pd.DataFrame({
    'mu': [mu],
    'sigma': [sigma],
    'nu': [nu],
    'Alpha': [alpha],
    'B1': [b1],
    'B2': [b2],
    'B3': [b3]
}).to_excel(output_file, index=False)

print(f"Saved to: {output_file}")