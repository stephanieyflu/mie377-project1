from services.strategies import *
import os
import pandas as pd
import numpy as np
from scipy.stats.mstats import gmean

# def project_function(periodReturns, periodFactRet, x0, p):
#     """
#     Please feel free to modify this function as desired
#     :param periodReturns:
#     :param periodFactRet:
#     :return: the allocation as a vector
#     """
#     Strategy3 = OLS_MVO_robust() # Vary parameters
#     x3 = Strategy3.execute_strategy(periodReturns, periodFactRet, alpha=0.95, llambda=1)

#     Strategy4 = PCA_MVO() # Vary parameters
#     x4 = Strategy4.execute_strategy(periodReturns, periodFactRet, p)

#     return x4

def project_function(periodReturns, periodFactRet, x0):
    """
    Please feel free to modify this function as desired
    :param periodReturns:
    :param periodFactRet:
    :return: the allocation as a vector
    """
    print('periodReturns:', periodReturns.shape)
    print('x0:', len(x0))
    # print(periodFactRet.columns)

    # Initialize all strategies
    x_equal = equal_weight(periodReturns)

    Strategy1 = HistoricalMeanVarianceOptimization()
    x_hist_mvo = Strategy1.execute_strategy(periodReturns, periodFactRet)

    Strategy2 = OLS_MVO()
    x_ols_mvo = Strategy2.execute_strategy(periodReturns, periodFactRet)

    Strategy3 = OLS_MVO_robust() # Vary parameters
    x_ols_mvo_robust = Strategy3.execute_strategy(periodReturns, periodFactRet, alpha=0.95, llambda=2)

    Strategy4 = PCA_MVO() # Vary parameters
    x_pca_mvo = Strategy4.execute_strategy(periodReturns, periodFactRet, p=7)

    Strategy5 = MARKET_CAP()
    x_market_cap = Strategy5.execute_strategy(periodReturns, periodFactRet)

    no_strat = 6

    strategy_weights = [x_equal, 
                        x_hist_mvo, 
                        x_ols_mvo, 
                        x_ols_mvo_robust, 
                        x_pca_mvo, 
                        x_market_cap]

    # Create DataFrame of all strategies
    strategies = pd.DataFrame([x_equal, 
                               x_hist_mvo, 
                               x_ols_mvo, 
                               x_ols_mvo_robust, 
                               x_pca_mvo, 
                               x_market_cap], 
                               columns=periodReturns.columns.values.tolist(),
                               index=['Equal Weight', 'Historical MVO', 
                                      'OLS MVO', 'OLS MVO Robust', 
                                      'PCA MVO', 'Market Cap Weights'])
    strategies['Sharpe Ratio'] = np.nan
    strategies['Turnover Rate'] = np.nan
    strategies['Score'] = np.nan
    strategies['Selected'] = 0

        # Check if the file exists - if not, we are in the calibration stage
    if not os.path.exists('portfolios_aes.csv'):
        # If the file does not exist, create it from the DataFrame
        strategies.to_csv('portfolios_aes.csv', index=False)
        print(f"File '{'portfolios_aes.csv'}' created.")

        penalties = pd.DataFrame(np.zeros(no_strat), columns=['Penalty'])
        penalties.to_csv('penalty_counter_aes.csv', index=False)

        selected = 0 # strategy 0
        strategies.at[selected-6, 'Selected'] = 1

        return strategy_weights[selected]
    
    else:
        # If the file exists, append the DataFrame to it
        strategies.to_csv('portfolios_aes.csv', mode='a', header=False, index=False)
        print(f"Data appended to '{'portfolios_aes.csv'}'.")

        strategies = pd.read_csv('portfolios_aes.csv')
        penalties = pd.read_csv('penalty_counter_aes.csv')
    
        # Calculate performance metrics
        sharpe_ratios = strategies.iloc[-no_strat*2:-no_strat, :-4].apply(lambda x: calculate_sharpe_ratio(x.values, periodReturns), axis=1)
        turnover_rates = strategies.iloc[-no_strat:, :-4].apply(lambda x: calculate_turnover_rate(x.values, x0), axis=1)
        scores = 0.8 * sharpe_ratios.values - 0.2 * turnover_rates.values

        # Subtract from the score the penalty
        for i in range(len(scores)):
            scores[i] -= penalties['Penalty'].values[i] * 0.05

        strategies.iloc[-6:, strategies.columns.get_loc('Sharpe Ratio')] = sharpe_ratios.values
        strategies.iloc[-6:, strategies.columns.get_loc('Turnover Rate')] = turnover_rates.values
        strategies.iloc[-6:, strategies.columns.get_loc('Score')] = scores

        selected = np.argmax(scores)
        strategies.at[selected-6, 'Selected'] = 1

        # Get the top half indices (rounded up)
        sorted_indices = np.argsort(scores)[::-1]
        top_half_indices = sorted_indices[:len(scores) // 2 + len(scores) % 2]
        
        prev_selected_vals = strategies.iloc[-2*no_strat:-no_strat]['Selected'].values
        prev_selected = np.where(prev_selected_vals == 1)[0][0]

        if prev_selected not in top_half_indices:
            penalties.at[prev_selected, 'Penalty'] += 1

        # Calculate penalties
        # If the strategy we selected last did not perform in the top half, add a penalty score to it

        strategies.to_csv('portfolios_aes.csv', mode='w', index=False)
        print(f"Data with scores updated to '{'portfolios_aes.csv'}'.")

        penalties.to_csv('penalty_counter_aes.csv', mode='w', index=False)

        return strategy_weights[selected]

# Define functions to calculate per period Sharpe ratio and turnover rate
def calculate_sharpe_ratio(weights, periodReturns, NumObs=36):
    '''
    Inputs:
        weights (np.ndarray): weights of current strategy being evaluated
        returns (np.ndarray): asset returns for the current period being evaluated
    
    Returns:
        sharpe_ratio (int)
    '''
    returns = periodReturns.iloc[(-1) * NumObs:, :]
    portfRets = pd.DataFrame(returns @ weights)
    sharpe_ratio = ((portfRets + 1).apply(gmean, axis=0) - 1)/portfRets.std()

    return sharpe_ratio.values[0]

def calculate_turnover_rate(x, x0):
    '''
    Inputs:
        x (np.ndarray): weights of current strategy being evaluated
        x0 (np.ndarray): weights of previous selected strategy
    
    Returns:
        turnover_rate (int)
    '''
    turnover_rate = np.mean(np.abs(x - x0)) 

    return turnover_rate
