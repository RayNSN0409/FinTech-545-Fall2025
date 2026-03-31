import numpy as np
import pandas as pd
from scipy.optimize import minimize

input_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test5_2.csv"
output_path = r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W13\testout10_1.csv"

def risk_parity_objective(weights, cov):
    n = len(weights)
    port_var = weights.T @ cov @ weights
    mrc = cov @ weights
    crc = weights * mrc
    rc_pct = crc / port_var
    target_rc = 1.0 / n
    return np.sum((rc_pct - target_rc)**2)

def get_risk_parity_weights(cov):
    n = cov.shape[0]
    init_guess = np.ones(n) / n
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = tuple((0.0, 1.0) for _ in range(n))
    
    result = minimize(
        risk_parity_objective, 
        init_guess, 
        args=(cov,), 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'ftol': 1e-12, 'disp': False}
    )
    return result.x

df_in = pd.read_csv(input_path)
cov_matrix = df_in.values

rp_weights = get_risk_parity_weights(cov_matrix)

df_out = pd.DataFrame(rp_weights, columns=['W'])
df_out.to_csv(output_path, index=False, float_format='%.9f')

print("W")
for w in rp_weights:
    print(f"{w:.9f}")