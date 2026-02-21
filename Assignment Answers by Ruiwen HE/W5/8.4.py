import pandas as pd
import numpy as np
from scipy.stats import norm

df = pd.read_csv(r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test7_1.csv")
x = df['x1'].values

mu = np.mean(x)
sigma = np.std(x, ddof=1)

alpha = 0.05

z = norm.ppf(alpha)
pdf_z = norm.pdf(z)

es_diff = sigma * pdf_z / alpha

es_abs = -mu + es_diff

out_df = pd.DataFrame({
    'ES Absolute': [es_abs], 
    'ES Diff from Mean': [es_diff]
})

out_df.to_csv(r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W5\testout_8.4.csv", index=False)
print(out_df)