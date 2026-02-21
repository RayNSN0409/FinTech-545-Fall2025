import pandas as pd
import numpy as np
from scipy.stats import t

df = pd.read_csv(r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test7_2.csv")
x = df['x1'].values

df_t, loc_t, scale_t = t.fit(x)

np.random.seed(42)      
n_simulations = 1000000  
alpha = 0.05

simulated_returns = t.rvs(df=df_t, loc=loc_t, scale=scale_t, size=n_simulations)

var_sim = np.percentile(simulated_returns, alpha * 100)

es_abs_sim = -np.mean(simulated_returns[simulated_returns <= var_sim])

es_diff_sim = es_abs_sim + loc_t

out_df = pd.DataFrame({
    'ES Absolute': [es_abs_sim], 
    'ES Diff from Mean': [es_diff_sim]
})

output_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W5\testout_8.6.csv"
out_df.to_csv(output_path, index=False)

print(out_df)

"""
--- Analysis & Conclusion ---

1. Observation: 
   The differences are extremely small (approx. -0.00006).

2. Cause of Difference (Monte Carlo Error): 
   Task 8.5 (Formula) calculates the exact theoretical integral. 
   Task 8.6 (Simulation) approximates this using a finite finite sample size. 
   Random sampling inherently introduces minor statistical noise.

3. Conclusion: 
   By the Law of Large Numbers, increasing the simulation size would shrink 
   this difference closer to zero. This proves the Monte Carlo method is 
   a highly accurate and reliable tool for estimating risk (ES), even without 
   complex analytical formulas.
"""