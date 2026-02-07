import pandas as pd
import numpy as np
file_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test1.csv"
df = pd.read_csv(file_path)
print(df.head())
cov_matrix_pairwise = df.cov()
print(cov_matrix_pairwise)
output_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W3\testout_1.3.csv"
cov_matrix_pairwise.to_csv(output_path)