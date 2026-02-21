import pandas as pd
import numpy as np
from scipy.stats import t

df = pd.read_csv(r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test7_2.csv")
x = df['x1'].values

df_t, loc_t, scale_t = t.fit(x)

alpha = 0.05

t_val = t.ppf(alpha, df_t)
pdf_t = t.pdf(t_val, df_t)

es_diff = scale_t * (pdf_t / alpha) * ((df_t + t_val**2) / (df_t - 1))

es_abs = -loc_t + es_diff

out_df = pd.DataFrame({
    'ES Absolute': [es_abs], 
    'ES Diff from Mean': [es_diff]
})

out_df.to_csv(r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W5\testout_8.5.csv", index=False)

print(out_df)