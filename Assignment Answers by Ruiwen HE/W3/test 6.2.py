import pandas as pd
import numpy as np

input_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test6.csv"
output_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W3\testout_6.2.csv"
df = pd.read_csv(input_path, index_col=0)

log_returns = np.log(df / df.shift(1)).dropna()

log_returns.to_csv(output_path)

print(log_returns.head())