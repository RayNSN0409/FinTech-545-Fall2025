import pandas as pd
import numpy as np
from scipy.stats import norm, t

portfolio = pd.read_csv(r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test9_1_portfolio.csv")
returns = pd.read_csv(r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\testfiles\data\test9_1_returns.csv")

info = {}
total_val = 0
for _, row in portfolio.iterrows():
    stock = row['Stock']
    val = row['Holding'] * row['Starting Price']
    dist = row['Distribution']
    info[stock] = {'Value': val, 'Dist': dist}
    total_val += val

fits = {}
u_returns = pd.DataFrame(index=returns.index)

for stock in returns.columns:
    dist_type = info[stock]['Dist']
    ret = returns[stock].values
    
    if dist_type == 'Normal':

        mu, std = norm.fit(ret)
        fits[stock] = {'mu': mu, 'std': std, 'type': 'Normal'}
  
        u_returns[stock] = norm.cdf(ret, loc=mu, scale=std)
        
    elif dist_type == 'T':

        df_t, loc_t, scale_t = t.fit(ret)
        fits[stock] = {'df': df_t, 'loc': loc_t, 'scale': scale_t, 'type': 'T'}
 
        u_returns[stock] = t.cdf(ret, df=df_t, loc=loc_t, scale=scale_t)

z_returns = pd.DataFrame(index=returns.index)
for stock in returns.columns:
    z_returns[stock] = norm.ppf(u_returns[stock])

corr_matrix = z_returns.corr().values

np.random.seed(42)  
n_simulations = 1000000  

z_sim = np.random.multivariate_normal([0, 0], corr_matrix, n_simulations)

u_sim = norm.cdf(z_sim)

simulated_returns = {}
stocks = list(returns.columns)
for i, stock in enumerate(stocks):
    dist_type = fits[stock]['type']
    if dist_type == 'Normal':
        simulated_returns[stock] = norm.ppf(u_sim[:, i], loc=fits[stock]['mu'], scale=fits[stock]['std'])
    elif dist_type == 'T':
        simulated_returns[stock] = t.ppf(u_sim[:, i], df=fits[stock]['df'], loc=fits[stock]['loc'], scale=fits[stock]['scale'])

pnl_sim = pd.DataFrame()
for stock in stocks:
    pnl_sim[stock] = simulated_returns[stock] * info[stock]['Value']

pnl_sim['Total'] = pnl_sim.sum(axis=1)

results = []
alpha = 0.05

for col in stocks + ['Total']:
    loss = -pnl_sim[col].values 

    var = np.percentile(loss, (1 - alpha) * 100)

    es = np.mean(loss[loss >= var])

    val = info[col]['Value'] if col in info else total_val
    var_pct = var / val
    es_pct = es / val
    
    results.append({
        'Stock': col,
        'VaR95': var,
        'ES95': es,
        'VaR95_Pct': var_pct,
        'ES95_Pct': es_pct
    })

results_df = pd.DataFrame(results)
print(results_df)

results_df.to_csv(r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\Assignment Answers by Ruiwen HE\W5\testout_9.1.csv", index=False)
