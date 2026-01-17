import pandas as pd
import os

# Setup paths
input_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test7_1.csv"
output_dir = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W2"

# Load data
df = pd.read_csv(input_path)
data = df.iloc[:, 0]

# Calculate Unbiased Estimates
# Pandas .std() uses ddof=1 (n-1) by default
mu = data.mean()
sigma = data.std()

# Print results
print(f"mu:    {mu:.6f}")
print(f"sigma: {sigma:.6f}")

# Save to Excel
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_file = os.path.join(output_dir, "normal_fit_results.xlsx")
pd.DataFrame({'mu': [mu], 'sigma': [sigma]}).to_excel(output_file, index=False)

print(f"Saved to: {output_file}")