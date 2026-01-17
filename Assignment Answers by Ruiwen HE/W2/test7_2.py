import pandas as pd
import os
from scipy.stats import t

# Input data path
input_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test7_2.csv"
df = pd.read_csv(input_path)

# Fit t-distribution (returns df/nu, loc/mu, scale/sigma)
nu, mu, sigma = t.fit(df.iloc[:, 0])

# Print results
print(f"mu:    {mu:.6f}")
print(f"sigma: {sigma:.6f}")
print(f"nu:    {nu:.6f}")

# Save results to Excel
output_dir = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W2"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_file = os.path.join(output_dir, "t_fit_results.xlsx")
pd.DataFrame({'mu': [mu], 'sigma': [sigma], 'nu': [nu]}).to_excel(output_file, index=False)

print(f"Results saved to: {output_file}")