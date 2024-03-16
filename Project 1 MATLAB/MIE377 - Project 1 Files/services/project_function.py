from services.strategies import *
import os
import pandas as pd

def project_function(periodReturns, periodFactRet, x0, alpha, llambda):
    """
    Please feel free to modify this function as desired
    :param periodReturns:
    :param periodFactRet:
    :return: the allocation as a vector
    """
    Strategy1 = OLS_MVO_robust() # Vary parameters
    x1 = Strategy1.execute_strategy(periodReturns, periodFactRet, alpha=alpha, llambda=llambda)

    Strategy2 = PCA_MVO() # Vary parameters
    x2 = Strategy2.execute_strategy(periodReturns, periodFactRet, p=4)

    return x1

# def project_function(periodReturns, periodFactRet, x0):
#     """
#     Please feel free to modify this function as desired
#     :param periodReturns:
#     :param periodFactRet:
#     :return: the allocation as a vector
#     """
#     print('periodReturns:', periodReturns.shape)
#     print('x0:', len(x0))
#     print(periodFactRet.columns)

#     # Initialize all strategies
#     x_equal = equal_weight(periodReturns)

#     Strategy1 = HistoricalMeanVarianceOptimization()
#     x_hist_mvo = Strategy1.execute_strategy(periodReturns, periodFactRet)

#     Strategy2 = OLS_MVO()
#     x_ols_mvo = Strategy2.execute_strategy(periodReturns, periodFactRet)

#     Strategy3 = OLS_MVO_robust() # Vary parameters
#     x_ols_mvo_robust = Strategy3.execute_strategy(periodReturns, periodFactRet, alpha=0.95, llambda=2)

#     Strategy4 = PCA_MVO() # Vary parameters
#     x_pca_mvo = Strategy4.execute_strategy(periodReturns, periodFactRet, p=7)

#     Strategy5 = MARKET_CAP()
#     x_market_cap = Strategy5.execute_strategy(periodReturns, periodFactRet)

#     no_strat = 6

#     # Create DataFrame of all strategies
#     strategies = pd.DataFrame([x_equal, 
#                                x_hist_mvo, 
#                                x_ols_mvo, 
#                                x_ols_mvo_robust, 
#                                x_pca_mvo, 
#                                x_market_cap], 
#                                columns=periodReturns.columns.values.tolist(),
#                                index=['Equal Weight', 'Historical MVO', 
#                                       'OLS MVO', 'OLS MVO Robust', 
#                                       'PCA MVO', 'Market Cap Weights'])

#     # Calculate performance metrics
#     performance_metrics = strategies.iloc[-no_strat:, :].apply(lambda x: calculate_performance_metrics(x, periodReturns, periodFactRet, x0), axis=1)

#     # Add column to .csv file with Sharpe ratio and turnover

#     # Select the strategy that has the best Sharpe ratio and average turnover

#     # Scores for Sharpe
#     scores = []
#     for group_idx, group_sharpe_ratios in strategies.groupby(strategies.index)['Sharpe Ratio']:
#         rank = group_sharpe_ratios.rank(method='dense', ascending=True) # higher score is better
#         score_mapping = {r: (no_strat+1) - r for r in rank}
#         scores.extend(rank.map(score_mapping))

#     df_sharpe = pd.DataFrame({'Sharpe Scores': scores}, index=strategies.index)
#     df_with_sharpe = pd.concat([strategies, df_sharpe], axis=1)

#     # Scores for turnover
#     scores = []
#     for group_idx, group_turnover in strategies.groupby(strategies.index)['Turnover']:
#         rank = group_turnover.rank(method='dense', ascending=False) # lower score is better
#         score_mapping = {r: (no_strat+1) - r for r in rank}
#         scores.extend(rank.map(score_mapping))

#     df_turnover = pd.DataFrame({'Turnover Scores': scores}, index=strategies.index)
#     df_with_sharpe_turnover = pd.concat([df_with_sharpe, df_turnover], axis=1)

#     # Add overall score
#     df_score = None
    
#     # Check if the file exists
#     if not os.path.exists('portfolios.csv'):
#         # If the file does not exist, create it from the DataFrame
#         strategies.to_csv('portfolios.csv')
#         print(f"File '{'portfolios.csv'}' created.")
#     else:
#         # If the file exists, append the DataFrame to it
#         strategies.to_csv('portfolios.csv', mode='a', header=False)
#         print(f"Data appended to '{'portfolios.csv'}'.")    

#     return x_equal

# def calculate_performance_metrics(strategy_weights, periodReturns, periodFactRet, x0):
#     # Calculate performance metrics
#     sharpe_ratio = calculate_sharpe_ratio(strategy_weights, periodReturns, periodFactRet, x0)
#     turnover_rate = calculate_turnover_rate(strategy_weights, x0)
    
#     # Return a tuple of performance metrics
#     return sharpe_ratio, turnover_rate

# # Define functions to calculate Sharpe ratio and turnover rate
# def calculate_sharpe_ratio(weights, returns, factors, x0):
    
#     return sharpe_ratio

# def calculate_turnover_rate(x, x0):
#     '''
#     Inputs:
#         x (np.ndarray): weights of current strategy being evaluated
#         x0 (np.ndarray): weights of previous selected strategy
    
#     Returns:
#         turnover_rate (int)
#     '''
#     turnover_rate = np.mean(np.abs(x - x0)) 
#     return turnover_rate
