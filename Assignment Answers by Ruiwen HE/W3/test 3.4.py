import pandas as pd
import numpy as np

input_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W3\testout_1.4.csv"
output_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W3\testout_3.4.csv"

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
corr_matrix = df.values

fixed_corr = higham_nearest(corr_matrix)

df_out = pd.DataFrame(fixed_corr, index=df.index, columns=df.columns)

print(df_out)
df_out.to_csv(output_path)