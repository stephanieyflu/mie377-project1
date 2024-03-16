from services.strategies import *


def project_function(periodReturns, periodFactRet, x0):
    """
    Please feel free to modify this function as desired
    :param periodReturns:
    :param periodFactRet:
    :return: the allocation as a vector
    """
    Strategy = OLS_MVO()
    StrategyC = OLS_CAPM()
    StrategyF = OLS_FF()
    StrategyCC = MVO_CC()
    StrategyL = Lasso_MVO()
    #x = equal_weight(periodReturns)
    x = StrategyL.execute_strategy(periodReturns, periodFactRet)
    return x


