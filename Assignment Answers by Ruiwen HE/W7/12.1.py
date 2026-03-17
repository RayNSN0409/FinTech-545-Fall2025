import numpy as np
import pandas as pd
from scipy.stats import norm

def calculate_gbsm_greeks(row):
    S = row['Underlying']
    K = row['Strike']
    T = row['DaysToMaturity'] / row['DayPerYear']
    r = row['RiskFreeRate']
    q = row['DividendRate']
    sigma = row['ImpliedVol']
    
    opt_type = str(row['Option Type']).strip().lower()
    
    b = r - q
    d1 = (np.log(S / K) + (b + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    exp_brT = np.exp((b - r) * T)
    exp_rT = np.exp(-r * T)
    sqrt_T = np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    
    if opt_type == 'call':
        value = S * exp_brT * norm.cdf(d1) - K * exp_rT * norm.cdf(d2)
        delta = exp_brT * norm.cdf(d1)
        rho = T * K * exp_rT * norm.cdf(d2)
        theta = -(S * exp_brT * pdf_d1 * sigma) / (2 * sqrt_T) - (b - r) * S * exp_brT * norm.cdf(d1) - r * K * exp_rT * norm.cdf(d2)
    elif opt_type == 'put':
        value = K * exp_rT * norm.cdf(-d2) - S * exp_brT * norm.cdf(-d1)
        delta = -exp_brT * norm.cdf(-d1)
        rho = -T * K * exp_rT * norm.cdf(-d2)
        theta = -(S * exp_brT * pdf_d1 * sigma) / (2 * sqrt_T) + (b - r) * S * exp_brT * norm.cdf(-d1) + r * K * exp_rT * norm.cdf(-d2)
    else:
        raise ValueError(f"Error!")

    gamma = (pdf_d1 * exp_brT) / (S * sigma * sqrt_T)
    vega = S * exp_brT * pdf_d1 * sqrt_T
    
    return pd.Series({
        'ID': int(row['ID']),
        'Value': value,
        'Delta': delta,
        'Gamma': gamma,
        'Vega': vega,
        'Rho': rho,
        'Theta': theta
    })

input_file = r'C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test12_1.csv'
df_input = pd.read_csv(input_file)
df_input = df_input.dropna(subset=['Option Type'])

df_output = df_input.apply(calculate_gbsm_greeks, axis=1)
df_output['ID'] = df_output['ID'].astype(int)
print(df_output)

output_file = r'C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W7\testout12_1.csv'
df_output.to_csv(output_file, index=False)