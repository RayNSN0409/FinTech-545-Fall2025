import pandas as pd
import numpy as np

input_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W3\testout_1.4.csv"
output_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W3\testout_3.2.csv"

def near_psd_corr(corr):

    vals, vecs = np.linalg.eigh(corr)

    vals = np.maximum(vals, 0)

    T = vecs @ np.diag(vals) @ vecs.T
    D_inv_sqrt = np.diag(1 / np.sqrt(np.diag(T)))
    return D_inv_sqrt @ T @ D_inv_sqrt

df = pd.read_csv(input_path, index_col=0)
corr_matrix = df.values

psd_corr = near_psd_corr(corr_matrix)

df_out = pd.DataFrame(psd_corr, index=df.index, columns=df.columns)
print(df_out)
df_out.to_csv(output_path)