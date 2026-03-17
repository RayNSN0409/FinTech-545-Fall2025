import numpy as np
import pandas as pd

def bt_american(call, underlying, strike, ttm, rf, b, ivol, N):
    dt = ttm / N
    u = np.exp(ivol * np.sqrt(dt))
    d = 1.0 / u
    pu = (np.exp(b * dt) - d) / (u - d)
    pd_prob = 1.0 - pu
    df = np.exp(-rf * dt)
    z = 1 if call else -1

    def idxFunc(i, j):
        return int(j * (j + 1) / 2) + i

    nNodes = int((N + 1) * (N + 2) / 2)
    optionValues = np.zeros(nNodes)

    for j in range(N, -1, -1):
        for i in range(j, -1, -1):
            idx = idxFunc(i, j)
            price = underlying * (u ** i) * (d ** (j - i))
            optionValues[idx] = max(0.0, z * (price - strike))
            
            if j < N:
                valNoExercise = df * (pu * optionValues[idxFunc(i + 1, j + 1)] + pd_prob * optionValues[idxFunc(i, j + 1)])
                optionValues[idx] = max(optionValues[idx], valNoExercise)

    return optionValues

def calculate_american_greeks(row):
    S, K = float(row['Underlying']), float(row['Strike'])
    T = float(row['DaysToMaturity']) / float(row['DayPerYear'])
    rf, q, ivol = float(row['RiskFreeRate']), float(row['DividendRate']), float(row['ImpliedVol'])
    
    call = True if str(row['Option Type']).strip().lower() == 'call' else False
    b, N = rf - q, 100 
    
    opt_vals = bt_american(call, S, K, T, rf, b, ivol, N)
    
    def idxFunc(i, j): return int(j * (j + 1) / 2) + i
        
    dt = T / N
    u, d = np.exp(ivol * np.sqrt(dt)), 1.0 / np.exp(ivol * np.sqrt(dt))
    
    C0, Cu, Cd = opt_vals[idxFunc(0, 0)], opt_vals[idxFunc(1, 1)], opt_vals[idxFunc(0, 1)]
    Cuu, Cud, Cdd = opt_vals[idxFunc(2, 2)], opt_vals[idxFunc(1, 2)], opt_vals[idxFunc(0, 2)]
    Su, Sd = S * u, S * d
    Suu, Sud, Sdd = S * u**2, S, S * d**2
    
    delta = (Cu - Cd) / (Su - Sd)
    gamma = (((Cuu - Cud) / (Suu - Sud)) - ((Cud - Cdd) / (Sud - Sdd))) / ((Suu - Sdd) / 2.0)
    theta = (C0 - Cud) / (2 * dt)
    
    d_vol, d_r = 1e-4, 1e-4
    vega = (bt_american(call, S, K, T, rf, b, ivol + d_vol, N)[0] - bt_american(call, S, K, T, rf, b, ivol - d_vol, N)[0]) / (2 * d_vol)
    rho = (bt_american(call, S, K, T, rf + d_r, b, ivol, N)[0] - bt_american(call, S, K, T, rf - d_r, b, ivol, N)[0]) / (2 * d_r)
    
    return pd.Series({
        'ID': int(row['ID']), 'Value': C0, 'Delta': delta, 
        'Gamma': gamma, 'Vega': vega, 'Rho': rho, 'Theta': theta
    })

input_file = r'C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test12_1.csv'
df_input = pd.read_csv(input_file).dropna(subset=['Option Type'])

df_output = df_input.apply(calculate_american_greeks, axis=1)
df_output['ID'] = df_output['ID'].astype(int)

print(df_output)
output_file = r'C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W7\testout12_2.csv'
df_output.to_csv(output_file, index=False)