import numpy as np
import pandas as pd

def bt_american_standard(call, underlying, strike, ttm, rf, b, ivol, N):
    if N == 0:
        return max(0.0, (1 if call else -1) * (underlying - strike))
        
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

    return optionValues[0]

def bt_american_discrete(call, underlying, strike, ttm, rf, divAmts, divTimes, ivol, N):
    if len(divAmts) == 0 or len(divTimes) == 0 or divTimes[0] > N:
        return bt_american_standard(call, underlying, strike, ttm, rf, rf, ivol, N)
    
    dt = ttm / N
    u = np.exp(ivol * np.sqrt(dt))
    d = 1.0 / u
    pu = (np.exp(rf * dt) - d) / (u - d)
    pd_prob = 1.0 - pu
    df = np.exp(-rf * dt)
    z = 1 if call else -1
    
    def idxFunc(i, j):
        return int(j * (j + 1) / 2) + i
        
    nNodes = int((divTimes[0] + 1) * (divTimes[0] + 2) / 2)
    optionValues = np.zeros(nNodes)
    
    for j in range(divTimes[0], -1, -1):
        for i in range(j, -1, -1):
            idx = idxFunc(i, j)
            price = underlying * (u ** i) * (d ** (j - i))
            
            if j < divTimes[0]:
                optionValues[idx] = max(0.0, z * (price - strike))
                valNoExercise = df * (pu * optionValues[idxFunc(i + 1, j + 1)] + pd_prob * optionValues[idxFunc(i, j + 1)])
                optionValues[idx] = max(optionValues[idx], valNoExercise)
            else:
                rem_divAmts = divAmts[1:]
                rem_divTimes = [t - divTimes[0] for t in divTimes[1:]]
                valNoExercise = bt_american_discrete(
                    call, price - divAmts[0], strike, 
                    ttm - divTimes[0] * dt, rf, 
                    rem_divAmts, rem_divTimes, ivol, N - divTimes[0]
                )
                valExercise = max(0.0, z * (price - strike))
                optionValues[idx] = max(valNoExercise, valExercise)
                
    return optionValues[0]

def calculate_discrete_dividends(row):
    S = float(row['Underlying'])
    K = float(row['Strike'])
    N = int(row['DaysToMaturity'])
    T = float(row['DaysToMaturity']) / float(row['DayPerYear'])
    rf = float(row['RiskFreeRate'])
    ivol = float(row['ImpliedVol'])
    
    opt_type = str(row['Option Type']).strip().lower()
    call = True if opt_type == 'call' else False
    
    div_dates_str = str(row['DividendDates'])
    div_amts_str = str(row['DividendAmts'])
    
    if pd.isna(row['DividendDates']) or div_dates_str.strip() == '':
        divTimes, divAmts = [], []
    else:
        divTimes = [int(float(x)) for x in div_dates_str.split(',')]
        divAmts = [float(x) for x in div_amts_str.split(',')]
        
    value = bt_american_discrete(call, S, K, T, rf, divAmts, divTimes, ivol, N)
    
    return pd.Series({
        'ID': row['ID'],
        'Value': value
    })

input_file = r'C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test12_3.csv'
df_input = pd.read_csv(input_file).dropna(subset=['Option Type'])

df_output = df_input.apply(calculate_discrete_dividends, axis=1)

df_output = df_output[['ID', 'Value']]
df_output['ID'] = df_output['ID'].astype(int)

print(df_output.to_string(index=False))

output_file = r'C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W7\testout12_3.csv'
df_output.to_csv(output_file, index=False)