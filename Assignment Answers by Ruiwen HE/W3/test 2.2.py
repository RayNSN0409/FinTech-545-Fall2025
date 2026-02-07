import pandas as pd
import numpy as np

input_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test2.csv"
output_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W3\testout_2.2.csv"

df = pd.read_csv(input_path)
df_clean = df.dropna()

X = df_clean.values
T, N = X.shape
lam = 0.94

weights = lam ** np.arange(T - 1, -1, -1)
weights /= weights.sum()

weighted_mean = (X * weights.reshape(-1, 1)).sum(axis=0)
X_centered = X - weighted_mean

cov_matrix = (X_centered.T * weights) @ X_centered

vol = np.sqrt(np.diag(cov_matrix))
corr_matrix = cov_matrix / np.outer(vol, vol)

df_corr = pd.DataFrame(corr_matrix, index=df_clean.columns, columns=df_clean.columns)
print(df_corr)
df_corr.to_csv(output_path)