from services.strategies import *
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def project_function(periodReturns, periodFactRet, x0):
    """
    :param periodReturns:
    :param periodFactRet:
    :param x0:
    :return: x (weight allocation as a vector)
    """
    Strategy = PCA_MVO() # Use the OLS_PCA_MVO strategy

    T, n = periodReturns.shape

    # Check if we are in the calibration period
    if T == 60: # 12 months * 5 years 
        # We are in the calibration period

        ##### Determine the optimal parameters #####

        ps = range(1, 11) # Range of p to test
        Ks = range(15, 21) # Range of K to test

        p_best, K_best, no_best = find_params(ps, Ks, Strategy, periodReturns, T)

        params = pd.DataFrame({'p': [p_best], 'K': [K_best], 'no': [no_best]})

        # print(params)

        if os.path.exists('params_aes.csv'):
            os.remove('params_aes.csv')
            # print("params_aes.csv deleted")
        
        params.to_csv('params_aes.csv', index=False)
        # print("best params saved in params_aes.csv")

        params = pd.read_csv('params_aes.csv')
        p_best = params.iloc[0, 0]
        K_best = params.iloc[0, 1]
        no_best = params.iloc[0, 2]

        x = Strategy.execute_strategy(periodReturns, NumObs=no_best, p=p_best, card=True, L=0.05, U=0.2, K=K_best)

    else: # No longer in the calibration period
        params = pd.read_csv('params_aes.csv') # Read the best parameters
        p_best = params.iloc[0, 0]
        K_best = params.iloc[0, 1]
        no_best = params.iloc[0, 2]

        x = Strategy.execute_strategy(periodReturns, NumObs=no_best, p=p_best, card=True, L=0.05, U=0.2, K=K_best)
    
        turnover = np.sum(np.abs(x-x0))
        if turnover > 1: # If turnover is high, use previous weights
            x = x0

    return x

def find_params(ps, Ks, Strategy, periodReturns, T):
    """Iterates through ps and Ks to determine the set of parameters that result in the optimal 
    Sharpe ratio during the calibration period

    Args:
        ps (np.ndarray): range of p to test
        Ks (np.ndarray): range of K to test
        Strategy (Class): strategy used to calculate portfolio weights
        periodReturns (pd.DataFrame): asset returns during the calibration period
        T (int): number of data points (i.e., observations) in periodReturns

    Returns:
        (best_p, best_K, best_no): best p, K, and NumObs parameters based on 
                                    Sharpe ratio during the calibration period
    """
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
                windowSize = w # NumObs

                numPeriods = (T - windowSize) // rebalancingFreq

                for t in range(numPeriods+1):
                    # Subset the returns and factor returns corresponding to the current calibration period
                    start_index = t * rebalancingFreq
                    end_index = t * rebalancingFreq + windowSize
                    
                    subperiodReturns = periodReturns.iloc[start_index:end_index]

                    if t > 0:
                        # Calculate the portfolio period returns
                        portfReturns.iloc[end_index-rebalancingFreq:end_index, portfReturns.columns.get_loc('Returns')] = subperiodReturns[-rebalancingFreq:].dot(weights)

                    weights = Strategy.execute_strategy(subperiodReturns, NumObs=w, p=p, card=True, L=0.05, U=0.2, K=K)
                
                # Calculate and save the Sharpe ratio for the current combination of parameters
                SR = (portfReturns.iloc[-(T-windowSize):]).mean() / (portfReturns.iloc[-(T-windowSize):]).std()
                SRs.append(SR[0])
                win_size.append(w)
                all_p.append(p)
                all_K.append(K)

    df = pd.DataFrame({'Window Size': win_size, 'p': all_p, 'K': all_K, 'SR': SRs})
    # plot(df, all_p)

    ##### Save the optimal parameters #####

    df_avg = df.groupby(['p', 'K'])['SR'].mean().reset_index()
    # df_avg.to_csv('aaa.csv')
    max_index = df_avg['SR'].idxmax()
    best_p = df_avg.at[max_index, 'p']
    best_K = df_avg.at[max_index, 'K']
    best_no = 48

    return best_p, best_K, best_no

def plot(df, all_p):
    """
    Plots Sharpe ratio with respect to p, K and NumObs during the calibration period
    """
    clrs = sns.color_palette('hls', n_colors=10)
    
    #### PLOT 1 #####
    fig, ax1 = plt.subplots()
    plt.title('Sharpe Ratio with respect to p and K for PCA-OLS-MVO-CC')
    plt.suptitle('NumObs = 24')
    ax1.set_xlabel('K')
    ax1.set_ylabel('Sharpe Ratio')

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
    plt.title('Sharpe Ratio with respect to p and K for PCA-OLS-MVO-CC')
    plt.suptitle('NumObs = 36')
    ax2.set_xlabel('K')
    ax2.set_ylabel('Sharpe Ratio')

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
    plt.title('Sharpe Ratio with respect to p and K for PCA-OLS-MVO-CC')
    plt.suptitle('NumObs = 48')
    ax3.set_xlabel('K')
    ax3.set_ylabel('Sharpe Ratio')

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
