import pandas as pd
import numpy as np

input_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W3\testout_1.3.csv"
output_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W3\testout_3.3.csv"

def projection_psd(A):
    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, 0)
    return vecs @ np.diag(vals) @ vecs.T

def projection_unit(A):
    A_out = A.copy()
    np.fill_diagonal(A_out, 1.0)
    return A_out

def higham_nearest(corr, tol=1e-9, max_iter=1000):
    delta_S = np.zeros_like(corr)
    Y = corr.copy()
    
    for k in range(max_iter):
        R = Y - delta_S
        X = projection_psd(R)
        delta_S = X - R
        Y_new = projection_unit(X)
        
        if np.linalg.norm(Y_new - Y, 'fro') < tol:
            break
        Y = Y_new
            
    return Y

df = pd.read_csv(input_path, index_col=0)
cov_matrix = df.values

std = np.sqrt(np.diag(cov_matrix))
inv_std = np.diag(1 / std)
corr_matrix = inv_std @ cov_matrix @ inv_std

fixed_corr = higham_nearest(corr_matrix)
fixed_cov = np.outer(std, std) * fixed_corr

df_out = pd.DataFrame(fixed_cov, index=df.index, columns=df.columns)

print(df_out)
df_out.to_csv(output_path)