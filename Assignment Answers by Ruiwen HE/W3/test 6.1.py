import pandas as pd

input_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test6.csv"
output_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W3\testout_6.1.csv"

df = pd.read_csv(input_path, index_col=0)

returns = df.pct_change().dropna()

returns.to_csv(output_path)

print(returns.head())