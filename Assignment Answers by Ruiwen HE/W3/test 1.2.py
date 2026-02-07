import pandas as pd
import numpy as np
file_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test1.csv"
df = pd.read_csv(file_path)
print(df.head())
df_clean = df.dropna()
corr_matrix = df_clean.corr()
print(corr_matrix)
output_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W3\testout_1.2.csv"
corr_matrix.to_csv(output_path)