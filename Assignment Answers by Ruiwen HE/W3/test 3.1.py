import pandas as pd
import numpy as np

input_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W3\testout_1.3.csv"
output_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W3\testout_3.1.csv"

def near_psd(A):
    std = np.sqrt(np.diag(A))
    inv_std = np.diag(1 / std)
    corr = inv_std @ A @ inv_std
    
    vals, vecs = np.linalg.eigh(corr)
    vals = np.maximum(vals, 0)
    
    T = vecs @ np.diag(vals) @ vecs.T
    T = np.diag(1 / np.sqrt(np.diag(T))) @ T @ np.diag(1 / np.sqrt(np.diag(T)))
    
    return np.outer(std, std) * T

df = pd.read_csv(input_path, index_col=0)
cov_matrix = df.values

psd_matrix = near_psd(cov_matrix)

df_out = pd.DataFrame(psd_matrix, index=df.index, columns=df.columns)
print(df_out)
df_out.to_csv(output_path)