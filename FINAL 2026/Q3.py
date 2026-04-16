#Using C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\FINAL 2026\problem2.csv:You have 3 assets to invest in. 6 month simulated returns for each asset are in problem2.csv. You plan to hold the assets for 6 months. The annual risk free rate is 4% (arithmetic annual compounding).Assume a budget constraint where the sum of weights = 1 for all optimizations
import pandas as pd
import numpy as np
#a. What is the holding period Sharpe Ratio of each asset? (5 pts)
df = pd.read_csv(r"C:\Users\RAYNSN\Desktop\QRM\FinTech-545-Fall2025\FINAL 2026\problem2.csv")
rf = 0.04  # Annual risk-free rate # notice that the data in csv is 6-month returns. since the question is about the holding period Sharpe Ratio, we should all calculate for 6-month period.
returns = df.mean()  # Average return for each asset
std_dev = df.std()  # Standard deviation for each asset
sharpe_ratios = (returns - rf/2) / std_dev  # Adjust risk-free rate for 6 months
print("Holding Period Sharpe Ratios for each asset:")
print(sharpe_ratios)






#b. Find the maximum Sharpe Ratio portfolio. Weights can be in [-1,1] (5 pts)
from scipy.optimize import minimize
def max_sharpe_ratio(returns, cov_matrix, rf):       
    num_assets = len(returns)
    args = (returns, cov_matrix, rf)
    
    def neg_sharpe_ratio(weights, returns, cov_matrix, rf):
        portfolio_return = np.dot(weights, returns)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(portfolio_return - rf) / portfolio_std
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = [(-1, 1) for _ in range(num_assets)]
    
    initial_guess = np.ones(num_assets) / num_assets
    result = minimize(neg_sharpe_ratio, initial_guess, args=args, constraints=constraints, bounds=bounds)
    
    return result.x # Optimal weights for maximum Sharpe Ratio portfolio
cov_matrix = df.cov()  # Covariance matrix of returns
max_sharpe_weights = max_sharpe_ratio(returns, cov_matrix, rf/2)  # Adjust risk-free rate for 6 months
print("\nOptimal Portfolio Weights for Maximum Sharpe Ratio:")
print(max_sharpe_weights)


#c. Explain the weights. Consider the correlations and Sharpe Ratios of the assets. (15 pts)
# answer: The weights of the maximum Sharpe Ratio portfolio indicate that the first asset has a negative weight, meaning it is being shorted, while the second and third assets have positive weights. This suggests that the first asset may have a lower Sharpe Ratio or higher volatility compared to the other two assets, leading to its short position in the optimal portfolio. The second asset has the highest weight, indicating it has the most favorable risk-return profile among the three assets. The third asset also has a positive weight, but it is less than the second asset, suggesting it has a lower Sharpe Ratio or higher volatility than the second asset. The correlations between the assets also play a role in determining the optimal weights, as they affect the overall portfolio risk. If the first asset is negatively correlated with the other two assets, it can help reduce portfolio risk when included in a short position.


#d. Now use the simulated values, and Expected Shortfall as the risk measure to find the optimal portfolio for (E(r)- rf) / ES. (10 pts)
def expected_shortfall(portfolio_returns, alpha=0.05):
    sorted_returns = np.sort(portfolio_returns)
    index = int(alpha * len(sorted_returns))
    return -np.mean(sorted_returns[:index])  # ES is typically reported as a positive number
def es_optimized_portfolio(returns, cov_matrix, rf):
    num_assets = len(returns)
    args = (returns, cov_matrix, rf)
    
    def neg_es_ratio(weights, returns, cov_matrix, rf):
        portfolio_return = np.dot(weights, returns)
        portfolio_returns = np.dot(df.values, weights)  # Simulated portfolio returns
        portfolio_es = expected_shortfall(portfolio_returns)
        return -(portfolio_return - rf) / portfolio_es
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = [(-1, 1) for _ in range(num_assets)]
    
    initial_guess = np.ones(num_assets) / num_assets
    result = minimize(neg_es_ratio, initial_guess, args=args, constraints=constraints, bounds=bounds)
    
    return result.x # Optimal weights for (E(r) - rf) / ES portfolio
print("\nOptimal Portfolio Weights for (E(r) - rf) / ES:")
es_weights = es_optimized_portfolio(returns, cov_matrix, rf/2)
print(es_weights)


#e. Discuss the weights. Why do they differ from the max Sharpe Ratio weights? (15 pts)
#Answer:  The weights of the optimal portfolio based on (E(r) - rf) / ES differ from the maximum Sharpe Ratio weights because they are optimized using a different risk measure. The maximum Sharpe Ratio portfolio focuses on maximizing the excess return per unit of standard deviation, which is a measure of volatility. In contrast, the portfolio optimized for (E(r) - rf) / ES focuses on maximizing the excess return per unit of Expected Shortfall, which is a measure of tail risk or extreme losses.

#f. Calculate the risk parity portfolio based on the simulation and ES. Report the weights and the Contribution to ES for each asset. (10 pts)
def risk_parity_portfolio(cov_matrix):
    num_assets = cov_matrix.shape[0]
    
    def objective(weights):
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        contributions = weights * np.dot(cov_matrix, weights) / portfolio_variance
        return np.sum((contributions - 1/num_assets)**2)
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = [(0, 1) for _ in range(num_assets)]
    
    initial_guess = np.ones(num_assets) / num_assets
    result = minimize(objective, initial_guess, constraints=constraints, bounds=bounds)
    
    return result.x # Risk parity portfolio weights
risk_parity_weights = risk_parity_portfolio(cov_matrix)
print("\nRisk Parity Portfolio Weights:")
print(risk_parity_weights)

print("\nContribution to ES for each asset in Risk Parity Portfolio:")
def contribution_to_es(weights, returns, cov_matrix):
    portfolio_es = expected_shortfall(np.dot(df.values, weights))
    contributions = weights * np.dot(cov_matrix, weights) / portfolio_es
    return contributions

contributions_es = contribution_to_es(risk_parity_weights, returns, cov_matrix)
print(contributions_es)


# g. Calculate the return and risk ex-ante attribution of the portfolio from part d. Compare to the return and risk exante attribution of the risk parity portfolio. Discuss. (15 pts)
def portfolio_return(weights, returns):
    return np.dot(weights, returns)
def portfolio_risk(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
def ex_ante_attribution(weights, returns, cov_matrix):
    port_return = portfolio_return(weights, returns)
    port_risk = portfolio_risk(weights, cov_matrix)
    return port_return, port_risk
print("\nEx-ante Attribution for Portfolio from Part D:")
es_port_return, es_port_risk = ex_ante_attribution(es_weights, returns, cov_matrix)
print(f"Expected Return: {es_port_return}, Risk (Std Dev): {es_port_risk}")
print("\nEx-ante Attribution for Risk Parity Portfolio:")
rp_port_return, rp_port_risk = ex_ante_attribution(risk_parity_weights, returns, cov_matrix)
print(f"Expected Return: {rp_port_return}, Risk (Std Dev): {rp_port_risk}")

print(-0.16050862 + 0.5384216  + 0.62208702)
