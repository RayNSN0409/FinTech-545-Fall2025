import pandas as pd 
import numpy as np
from scipy.stats import norm

# 1. Load Data
file_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test7_1.csv"
df = pd.read_csv(file_path)

# 2. Calculate key statistics
mu = df.mean()
sigma = df.std()
alpha = 1 - 0.95
z_score = norm.ppf(alpha)

# 3. Calculate VaR
VaR_95 = mu + z_score * sigma
VaR_95_abs = abs(VaR_95)
diff = abs(mu - VaR_95)

# 4. Save to CSV
output_df = pd.DataFrame({
    'VaR Absolute': [VaR_95_abs.values[0]],
    'VaR Diff from Mean': [diff.values[0]]
})
print (output_df.to_string(index=False))
output_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W4\testout_8.1.csv"

# Save File
# Ensure the directory exists or handle exception if needed
output_df.to_csv(output_path, index=False)