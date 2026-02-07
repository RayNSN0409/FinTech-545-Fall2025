import pandas as pd
import numpy as np

input_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test2.csv"
output_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W3\testout_2.3.csv"

df = pd.read_csv(input_path)
df_clean = df.dropna()
X = df_clean.values
T, N = X.shape

def get_ew_cov(matrix, lam):
    weights = lam ** np.arange(len(matrix) - 1, -1, -1)
    weights /= weights.sum()
    
    weighted_mean = (matrix * weights.reshape(-1, 1)).sum(axis=0)
    matrix_centered = matrix - weighted_mean

    return (matrix_centered.T * weights) @ matrix_centered

cov_97 = get_ew_cov(X, 0.97)
std_97 = np.sqrt(np.diag(cov_97)) 

cov_94 = get_ew_cov(X, 0.94)
std_94 = np.sqrt(np.diag(cov_94))
corr_94 = cov_94 / np.outer(std_94, std_94) 
final_cov = corr_94 * np.outer(std_97, std_97)

df_final = pd.DataFrame(final_cov, index=df_clean.columns, columns=df_clean.columns)
print(df_final)
df_final.to_csv(output_path)