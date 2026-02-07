import pandas as pd
import numpy as np

input_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W3\testout_3.1.csv"
output_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W3\testout_4.1.csv"

def chol_psd(root):
    n = root.shape[0]
    L = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1):
            s = np.dot(L[i, :j], L[j, :j])
            
            if i == j:
                L[i, j] = np.sqrt(max(root[i, i] - s, 0))
            else:
                if L[j, j] > 0:
                    L[i, j] = (root[i, j] - s) / L[j, j]
                else:
                    L[i, j] = 0
                    
    return L

df = pd.read_csv(input_path, index_col=0)
cov_matrix = df.values

chol_matrix = chol_psd(cov_matrix)

df_out = pd.DataFrame(chol_matrix, index=df.index, columns=df.columns)

print(df_out)   
df_out.to_csv(output_path)