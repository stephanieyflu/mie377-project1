import numpy as np
from services.estimators import *
from services.optimization import *
import pandas as pd

# this file will produce portfolios as outputs from data - the strategies can be implemented as classes or functions
# if the strategies have parameters then it probably makes sense to define them as a class

class PCA_MVO:
    """
    uses PCA to estimate the covariance matrix and expected return
    and MVO with cardinality constraints
    """

    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, NumObs=36, p=3, robust=False, T=None, alpha=None, llambda=None, card=False, L=0.3, U=1, K=10):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param factorReturns:
        :param periodReturns:
        :param NumObs:
        :param p: number of PCs to select as factors
        :param robust:
        :param T:
        :param alpha:
        :param llambda:
        :param card:
        :param L:
        :param U:
        :param K:
        :return: x
        """
        T, n = periodReturns.shape
        # get the last T observations
        returns = periodReturns.iloc[(-1) * NumObs:, :]
        mu, Q = PCA(returns, p=p)
        x = MVO(mu, Q, robust=robust, T=T, alpha=alpha, llambda=llambda, card=card, L=L, U=U, K=K)
        return x


class MARKET_CAP:
    """
    uses an estimate of the market portfolio weights as the portfolio
    """

    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use
    
    def execute_strategy(self, periodReturns, factorReturns):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param factorReturns:
        :param periodReturns:
        :return: x
        """
        T, n = periodReturns.shape
        # get the last T observations
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        x = market_cap(factRet['Mkt_RF'].values, returns.values)

        return x
    

def equal_weight(periodReturns):
    """
    computes the equal weight vector as the portfolio
    :param periodReturns:
    :return: x
    """
    T, n = periodReturns.shape
    x = (1 / n) * np.ones([n])
    return x


class HistoricalMeanVarianceOptimization:
    """
    uses historical returns to estimate the covariance matrix and expected return
    """

    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns=None):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param periodReturns:
        :param factorReturns:
        :return: x
        """
        factorReturns = None  # we are not using the factor returns
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        mu = np.expand_dims(returns.mean(axis=0).values, axis=1)
        Q = returns.cov().values
        x = MVO(mu, Q)

        return x


class OLS_MVO:
    """
    uses all 8 factors to estimate the covariance matrix and expected return
    and regular MVO
    """

    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param factorReturns:
        :param periodReturns:
        :return: x
        """
        T, n = periodReturns.shape
        # get the last T observations
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        mu, Q = OLS(returns, factRet)
        x = MVO(mu, Q)
        return x


class OLS_CAPM:
    """
    uses the market factor to estimate the covariance matrix and expected return
    and regular MVO
    """

    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param factorReturns:
        :param periodReturns:
        :return: x
        """
        T, n = periodReturns.shape
        # get the last T observations
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        mu, Q = OLS(returns, factRet[["Mkt_RF"]])
        x = MVO(mu, Q)
        return x
    

class OLS_FF:
    """
    uses the Fama-French factors to estimate the covariance matrix and expected return
    and regular MVO
    """

    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param factorReturns:
        :param periodReturns:
        :return: x
        """
        T, n = periodReturns.shape
        # get the last T observations
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        mu, Q = OLS(returns, factRet[["Mkt_RF", "SMB", "HML"]])
        x = MVO(mu, Q)
        return x
    

class Lasso_MVO:
    """
    uses LASSO to select factors to use to estimate the covariance matrix and expected return
    and regular MVO
    """

    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns, S=0.001):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param factorReturns:
        :param periodReturns:
        :param S:
        :return: x
        """
        T, n = periodReturns.shape
        # get the last T observations
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        mu, Q = Lasso(returns, factRet, S)
        x = MVO(mu, Q)
        return x

    
class MVO_CC:
    """
    adds cardinality constraints and buy-in thresholds to regular MVO
    """

    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns, L=0.03, U=1, K=10):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param factorReturns:
        :param periodReturns:
        :param L:
        :param U:
        :param K:
        :return: x
        """
        T, n = periodReturns.shape
        # get the last T observations
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        mu, Q = OLS(returns, factRet)
        x = MVO(mu, Q, card=True, L=L, U=U, K=K)
        return x
      

class OLS_MVO_robust:
    """
    uses all 8 factors to estimate the covariance matrix and expected return
    and robust MVO with an ellipsoidal uncertainty set
    """

    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def find_lambda():
        
        return

    def execute_strategy(self, periodReturns, factorReturns, alpha=0.95, llambda=10):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param factorReturns:
        :param periodReturns:
        :param alpha: alpha value for robust MVO 
        :param llambda: lambda value for robust MVO
        :return: x
        """
        T, n = periodReturns.shape
        # get the last T observations
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        mu, Q = OLS(returns, factRet)
        x = MVO(mu, Q, robust=True, T=T, alpha=alpha, llambda=llambda)
        return x


class Max_Sharpe_Ratio:
    def __init__(self, NumObs = 36):
        self.NumObs = NumObs # number of observations to use
    
    def execute_strategy(self, periodReturns, factorReturns):
       T, n = periodReturns.shape
       returns = periodReturns.iloc[(-1) * self.NumObs:, :]
       factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
       mu, Q = OLS(returns, factRet)
       x = Sharpe_MVO(mu, Q)
       return x
    

class Max_Sharpe_Robust_Ratio: # NOT USING
    def __init__(self, NumObs = 36):
        self.NumObs = NumObs # number of observations to use
    
    def execute_strategy(self, periodReturns, factorReturns):
       T, n = periodReturns.shape
       returns = periodReturns.iloc[(-1) * self.NumObs:, :]
       factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
       mu, Q = OLS(returns, factRet)
       x = Robust_Sharpe_Eps_MVO(mu, Q)
       return x
    

class Strat_Max_Sharpe_Min_Turn:
    def __init__(self, NumObs = 36):
        self.NumObs = NumObs # number of observations to use
    
    def execute_strategy(self, periodReturns, factorReturns, x0, llambda=1):
       T, n = periodReturns.shape
       returns = periodReturns.iloc[(-1) * self.NumObs:, :]
       factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
       mu, Q = OLS(returns, factRet)
       x = Max_Sharpe_Min_Turn(mu, Q, x0, llambda)
       return x