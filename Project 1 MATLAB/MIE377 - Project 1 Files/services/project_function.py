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

    # Check if we are in the calibration period
    T, n = periodReturns.shape

    if T == 60: # 12 months * 5 years 
        # We are in the calibration period

        ##### Determine the optimal parameters #####

        ps = range(1, 11)
        Ks = range(15, 21)

        p_best, K_best, no_best = find_params(ps, Ks, Strategy, periodReturns, T)

        params = pd.DataFrame({'p': [p_best], 'K': [K_best], 'no': [no_best]})

        print(params)

        if os.path.exists('params_aes.csv'):
            os.remove('params_aes.csv')
            print("params_aes.csv deleted")
        
        params.to_csv('params_aes.csv', index=False)
        print("best params saved in params_aes.csv")

        params = pd.read_csv('params_aes.csv')
        p_best = params.iloc[0, 0]
        K_best = params.iloc[0, 1]
        no_best = params.iloc[0, 2]

        x = Strategy.execute_strategy(periodReturns, NumObs=no_best, p=p_best, card=True, L=0.05, U=0.225, K=K_best)

    else: # only care about turnover if we're not during calibration
        params = pd.read_csv('params_aes.csv')
        p_best = params.iloc[0, 0]
        K_best = params.iloc[0, 1]
        no_best = params.iloc[0, 2]

        x = Strategy.execute_strategy(periodReturns, NumObs=no_best, p=p_best, card=True, L=0.05, U=0.225, K=K_best)
    
        turnover = np.sum(np.abs(x-x0))
        if turnover > 1: # justify this value!!
            print('large turnover')
            x = x0

    return x

def find_params(ps, Ks, Strategy, periodReturns, T):
    SRs = []
    win_size = []
    all_p = []
    all_K = []

    for w in [24, 36, 48]: # this takes way too long for more stocks - try just 36 and 48 or just 48?
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

                    if t > 0:
                        # print(t)
                        # print(end_index)
                        portfReturns.iloc[end_index-rebalancingFreq:end_index, portfReturns.columns.get_loc('Returns')] = subperiodReturns[-rebalancingFreq:].dot(weights)

                    weights = Strategy.execute_strategy(subperiodReturns, NumObs=w, p=p, card=True, L=0.05, U=0.225, K=K)
                
                SR = (portfReturns.iloc[-(T-windowSize):]).mean() / (portfReturns.iloc[-(T-windowSize):]).std()
                SRs.append(SR[0])
                win_size.append(w)
                all_p.append(p)
                all_K.append(K)

    df = pd.DataFrame({'Window Size': win_size, 'p': all_p, 'K': all_K, 'SR': SRs})

    plot(df, all_p)

    ##### Save the optimal parameters #####

    # # *** Method 1 *** #
    # max_index = df['SR'].idxmax()
    # best_p = df.at[max_index, 'p']
    # best_K = df.at[max_index, 'K']
    # best_no = df.at[max_index, 'Window Size']

    # *** Method 2 *** # - best results so far when we have avg over 24/36 to 48 (just over 48 is not bad though)
    df_avg = df.groupby(['p', 'K'])['SR'].mean().reset_index()
    df_avg.to_csv('aaa.csv')
    max_index = df_avg['SR'].idxmax()
    best_p = df_avg.at[max_index, 'p']
    best_K = df_avg.at[max_index, 'K']
    # best_no = df.at[max_index, 'Window Size']
    best_no = 48
    
    # # *** Method 3 *** #
    # df_avg = df.groupby(['p', 'K'])['SR'].mean().reset_index()
    # df_avg.to_csv('aaa.csv')
    # best_p = df_avg.at[len(Ks)-1, 'p']
    # best_K = 10
    # best_no = 48

    return best_p, best_K, best_no

def plot(df, all_p):
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