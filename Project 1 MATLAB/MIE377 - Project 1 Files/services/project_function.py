from services.strategies import *
import os
import pandas as pd


def project_function(periodReturns, periodFactRet, x0):
    """
    Please feel free to modify this function as desired
    :param periodReturns:
    :param periodFactRet:
    :return: the allocation as a vector
    """
    Strategy = Strat_Max_Sharpe_Min_Turn_1()
    #x = equal_weight(periodReturns)
    x = Strategy.execute_strategy(periodReturns, periodFactRet, x0)
    return x
    '''
    Strategy1 = Max_Sharpe_Low_Turn_Ratio()
    x_sharpe = Strategy1.execute_strategy(periodReturns, periodFactRet, x0)

    Strategy2 = OLS_MVO()
    x_mvo = Strategy2.execute_strategy(periodReturns, periodFactRet)

    strategies = pd.DataFrame([x_mvo, x_sharpe], index=['MVO', 'Sharpe'])

    # Check if the file exists
    if not os.path.exists('portfolios.csv'):
        # If the file does not exist, create it from the DataFrame
        strategies.to_csv('portfolios.csv', index=False)
        print(f"File '{'portfolios.csv'}' created.")
    else:
        # If the file exists, append the DataFrame to it

        strategies.to_csv('portfolios.csv', mode='a', header=False, index=False)
        print(f"Data appended to '{'portfolios.csv'}'.")

    # introduce logic/model to select the portfolio from how the past portfolios did
    x = x_mvo
    return x

#label different strate i.e
#strate1 = mbo
#strat2 = low sharpe
'''