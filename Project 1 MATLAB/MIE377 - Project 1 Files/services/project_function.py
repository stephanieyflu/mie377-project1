from services.strategies import *
import os
import pandas as pd
import numpy as np
from scipy.stats.mstats import gmean
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

def project_function(periodReturns, periodFactRet, x0):
    """
    Please feel free to modify this function as desired
    :param periodReturns:
    :param periodFactRet:
    :return: the allocation as a vector
    """
    Strategy = PCA_MVO() # Vary parameters
    # x = Strategy.execute_strategy(periodReturns, NumObs=36, p=7, card=True, L=0.05, U=1, K=6)
    # params = pd.read_csv('params_aes.csv')
    # p_best = params.iloc[0, 0]
    # K_best = params.iloc[0, 1] + 5
    # no_best = params.iloc[0, 2]

    # x = Strategy.execute_strategy(periodReturns, NumObs=no_best, p=p_best, card=True, L=0.05, U=1, K=K_best)

    # Check if we are in the calibration period
    T, n = periodReturns.shape

    if T == 60: # 12 months * 5 years 
        # We are in the calibration period

        ##### Determine the optimal parameters #####

        ps = range(1, 11) # i
        Ks = range(5, 16) # j

        p_best, K_best, no_best = find_params(ps, Ks, Strategy, periodReturns, periodFactRet, T)

        params = pd.DataFrame({'p': [p_best], 'K': [K_best], 'no': [no_best]})

        if os.path.exists('params_aes.csv'):
            os.remove('params_aes.csv')
            print("params_aes.csv deleted")
        
        params.to_csv('params_aes.csv', index=False)
        print("best params saved in params_aes.csv")

        params = pd.read_csv('params_aes.csv')
        p_best = params.iloc[0, 0]
        K_best = params.iloc[0, 1] + 5
        no_best = params.iloc[0, 2]

        x = Strategy.execute_strategy(periodReturns, NumObs=no_best, p=p_best, card=True, L=0.05, U=1, K=K_best)

    else: # only care about turnover if we're not during calibration
        params = pd.read_csv('params_aes.csv')
        p_best = params.iloc[0, 0]
        K_best = params.iloc[0, 1] + 5
        no_best = params.iloc[0, 2]

        x = Strategy.execute_strategy(periodReturns, NumObs=no_best, p=p_best, card=True, L=0.05, U=1, K=K_best)
    
        # turnover = np.sum(np.abs(x-x0))
        # if turnover < 0.5: # justify this value!!
        #     x = x0

    return x

def find_params(ps, Ks, Strategy, periodReturns, periodFactRet, T):
    SRs = []
    win_size = []
    all_p = []
    all_K = []

    for w in [24, 36, 48]:
        for p in ps:
            for K in Ks:

                # Preallocate space for the portfolio per period value and turnover
                portfReturns = pd.DataFrame({'Returns': np.zeros(T)}, index=periodReturns.index)

                rebalancingFreq = 6
                windowSize = w

                numPeriods = (T - windowSize) // rebalancingFreq

                for t in range(numPeriods+1):
                    # Subset the returns and factor returns corresponding to the current calibration period.
                    start_index = t * rebalancingFreq
                    end_index = t * rebalancingFreq + windowSize
                    
                    subperiodReturns = periodReturns.iloc[start_index:end_index]
                    subperiodFactRet = periodFactRet.iloc[start_index:end_index]

                    if t > 0:
                        # print(t)
                        # print(end_index)
                        portfReturns.iloc[end_index-rebalancingFreq:end_index, portfReturns.columns.get_loc('Returns')] = subperiodReturns[-rebalancingFreq:].dot(weights)

                    weights = Strategy.execute_strategy(subperiodReturns, NumObs=w, p=p, card=True, L=0.05, U=1, K=K)
                
                SR = (portfReturns.iloc[-(T-windowSize):]).mean() / (portfReturns.iloc[-(T-windowSize):]).std()
                SRs.append(SR[0])
                win_size.append(w)
                all_p.append(p)
                all_K.append(K)

    df = pd.DataFrame({'Window Size': win_size, 'p': all_p, 'K': all_K, 'SR': SRs})

    ##### PLOT 1 #####
    fig, ax1 = plt.subplots()
    plt.title('Sharpe Ratio with respect to p and K for PCA-OLS-MVO-CC')
    ax1.set_xlabel('K')
    ax1.set_ylabel('Sharpe Ratio')

    clrs = sns.color_palette('hls', n_colors=15)

    df1 = df[df['Window Size'] == 24]
    i = 0
    for p in sorted(list(set(all_p))):
        sub_df = df1[df1['p'] == p]
        # display(sub_df)
        ax1.plot(sub_df['K'].values, sub_df['SR'].values, label=str(p), marker='.', color=clrs[i])
        i += 1

    ax1.legend(title='p', bbox_to_anchor=(1.2, 1.03))
    plt.tight_layout()
    plt.show()

    ##### PLOT 2 #####
    fig, ax2 = plt.subplots()

    df1 = df[df['Window Size'] == 36]
    i = 0
    for p in sorted(list(set(all_p))):
        sub_df = df1[df1['p'] == p]
        # display(sub_df)
        ax2.plot(sub_df['K'].values, sub_df['SR'].values, label=str(p), marker='.', color=clrs[i])
        i += 1

    ax2.legend(title='p', bbox_to_anchor=(1.2, 1.03))
    plt.tight_layout()
    plt.show()

    ##### PLOT 3 #####
    fig, ax3 = plt.subplots()

    df1 = df[df['Window Size'] == 48]
    i = 0
    for p in sorted(list(set(all_p))):
        sub_df = df1[df1['p'] == p]
        # display(sub_df)
        ax3.plot(sub_df['K'].values, sub_df['SR'].values, label=str(p), marker='.', color=clrs[i])
        i += 1

    ax3.legend(title='p', bbox_to_anchor=(1.2, 1.03))
    plt.tight_layout()
    plt.show()

    df.to_csv('abc.csv')

    ##### Save the optimal parameters #####
    max_index = df['SR'].idxmax()

    best_p = df.at[max_index, 'p']
    best_K = df.at[max_index, 'K']
    best_no = df.at[max_index, 'Window Size']

    return best_p, best_K, best_no

# def project_function(periodReturns, periodFactRet, x0):
#     """
#     Please feel free to modify this function as desired
#     :param periodReturns:
#     :param periodFactRet:
#     :return: the allocation as a vector
#     """
#     # if x0 == 0, then we know that we're at the beginning of a test and we need a new file
#     if np.product(x0 == 0):
#         if os.path.exists('portfolios_aes.csv'):
#             os.remove('portfolios_aes.csv')
#             print("portfolios_aes.csv deleted")
#         if os.path.exists('penalty_counter_aes.csv'):
#             os.remove('penalty_counter_aes.csv')
#             print("penalty_counter_aes.csv deleted")

#     ##### Define strategy names #####
#     names = ['Historical MVO', 
#              'OLS MVO', 
#              'OLS MVO Robust', 
#              'PCA MVO', 
#              'Market Cap',
#              'Equal Weight']

#     ##### Calculate weights for all strategies #####
#     # x1 = equal_weight(periodReturns)
#     # x2 = OLS_CAPM().execute_strategy(periodReturns, periodFactRet)
#     # x3 = OLS_FF().execute_strategy(periodReturns, periodFactRet)
#     Strategy2 = Strat_Max_Sharpe_Min_Turn()
#     x1 = Strategy2.execute_strategy(periodReturns, periodFactRet, x0, llambda=2)

#     # Strategy 1
#     # Strategy1 = Max_Sharpe_Ratio()
#     # x1 = Strategy1.execute_strategy(periodReturns, periodFactRet)
#     # x1 = x0
#     # Strategy1 = OLS_FF()
#     # x1 = Strategy1.execute_strategy(periodReturns, periodFactRet)

#     # # Strategy 2
#     # Strategy2 = Strat_Max_Sharpe_Min_Turn()
#     # x2 = Strategy2.execute_strategy(periodReturns, periodFactRet, x0)

#     # # Strategy 3
#     # Strategy3 = HistoricalMeanVarianceOptimization()
#     # x3 = Strategy3.execute_strategy(periodReturns, periodFactRet)

#     # # Strategy 4
#     # Strategy4 = OLS_MVO()
#     # x4 = Strategy4.execute_strategy(periodReturns, periodFactRet)

#     # # Strategy 5
#     # Strategy5 = OLS_MVO_robust() # Vary parameters
#     # x5 = Strategy5.execute_strategy(periodReturns, periodFactRet, alpha=0.95, llambda=2)

#     # # Strategy 6
#     # Strategy6 = PCA_MVO() # Vary parameters
#     # x6 = Strategy6.execute_strategy(periodReturns, periodFactRet, p=7)

#     # # Strategy 7
#     # Strategy7 = MARKET_CAP()
#     # x7 = Strategy7.execute_strategy(periodReturns, periodFactRet)

#     # # Strategy 8
#     # x8 = equal_weight(periodReturns)

#     no_strat = 1
#     strategy_weights = [x1]

#     # Create DataFrame of all strategy weights
#     strategies = pd.DataFrame([x1], 
#                                columns=periodReturns.columns.values.tolist())
#     strategies['Sharpe Ratio'] = np.nan
#     strategies['Turnover Rate'] = np.nan
#     strategies['Score'] = np.nan
#     strategies['Selected'] = 0

#     # Check if the file exists - if not, we are in the calibration stage
#     if not os.path.exists('portfolios_aes.csv'):
#         selected = 0 # strategy 1 - find a better way to use the calibration period to select an initial model?
#         strategies.at[strategies.index[selected-no_strat], 'Selected'] = 1

#         # If the file does not exist, create it from the DataFrame
#         strategies.to_csv('portfolios_aes.csv', index=False)
#         print(f"File '{'portfolios_aes.csv'}' created.")

#         penalties = pd.DataFrame(np.zeros(no_strat), columns=['Penalty'])
#         penalties.to_csv('penalty_counter_aes.csv', index=False)
#         print(f"Penalties file created.")
    
#     else:
#         # If the file exists, append the DataFrame to it
#         strategies.to_csv('portfolios_aes.csv', mode='a', header=False, index=False)
#         print(f"Data appended to '{'portfolios_aes.csv'}'.")

#         strategies = pd.read_csv('portfolios_aes.csv')
#         penalties = pd.read_csv('penalty_counter_aes.csv')
    
#         # Calculate performance metrics
#         sharpe_ratios = strategies.iloc[-no_strat*2:-no_strat, :-4].apply(lambda x: calculate_sharpe_ratio(x.values, periodReturns), axis=1)
#         turnover_rates = strategies.iloc[-no_strat:, :-4].apply(lambda x: calculate_turnover_rate(x.values, x0), axis=1)
#         sharpe_ratios.fillna(0, inplace=True)

#         # Normalize the Sharpe ratios and turnover rates
#         scaler = MinMaxScaler()
#         sharpe_normalized = scaler.fit_transform(sharpe_ratios.values.reshape(-1, 1))
#         turnover_normalized = scaler.fit_transform(turnover_rates.values.reshape(-1, 1))
        
#         # Calculate the initial score for each strategy
#         scores = 0.8 * sharpe_normalized - 0.2 * turnover_normalized

#         # Subtract each strategy's penalty for its initial score
#         # for i in range(len(scores)):
#         #     scores[i] -= penalties['Penalty'].values[i] * 0.1

#         # Add scores to the DataFrame
#         strategies.iloc[-no_strat:, strategies.columns.get_loc('Sharpe Ratio')] = sharpe_ratios.values
#         strategies.iloc[-no_strat:, strategies.columns.get_loc('Turnover Rate')] = turnover_rates.values
#         strategies.iloc[-no_strat:, strategies.columns.get_loc('Score')] = scores

#         # Determine the best strategy
#         selected = np.argmax(scores)
#         strategies.at[strategies.index[selected-no_strat], 'Selected'] = 1

#         # Assign new penalty values
#         # Get the top half indices (rounded up)
#         sorted_indices = np.argsort(scores)[::-1]
#         top_half_indices = sorted_indices[:len(scores) // 2 + len(scores) % 2]
        
#         prev_selected_vals = strategies.iloc[-2*no_strat:-no_strat, strategies.columns.get_loc('Selected')].values
#         prev_selected = np.where(prev_selected_vals == 1)[0][0]

#         if prev_selected not in top_half_indices:
#             penalties.at[penalties.index[prev_selected], 'Penalty'] += 1

#         # Update csv files with data from this iteration
#         strategies.to_csv('portfolios_aes.csv', mode='w', index=False)
#         print(f"Data with scores updated to '{'portfolios_aes.csv'}'.")

#         penalties.to_csv('penalty_counter_aes.csv', mode='w', index=False)
#         print(f"Penalties updated.")

#     print("########### {} Selected ###########\n".format(names[selected]))
#     return strategy_weights[selected]


# def calculate_sharpe_ratio(weights, periodReturns, NumObs=36):
#     '''
#     Inputs:
#         weights (np.ndarray): weights of current strategy being evaluated
#         returns (np.ndarray): asset returns for the current period being evaluated
    
#     Returns:
#         sharpe_ratio (int)
#     '''
#     returns = periodReturns.iloc[(-1) * NumObs:, :]
#     portfRets = pd.DataFrame(returns @ weights)
#     sharpe_ratio = ((portfRets + 1).apply(gmean, axis=0) - 1)/portfRets.std()

#     return sharpe_ratio.values[0]


# def calculate_turnover_rate(x, x0):
#     '''
#     Inputs:
#         x (np.ndarray): weights of current strategy being evaluated
#         x0 (np.ndarray): weights of previous selected strategy
    
#     Returns:
#         turnover_rate (int)
#     '''
#     turnover_rate = np.sum(np.abs(x - x0)) 

#     return turnover_rate
