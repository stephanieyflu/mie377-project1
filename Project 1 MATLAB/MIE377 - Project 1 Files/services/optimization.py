import cvxpy as cp
import numpy as np
from scipy.stats import chi2

def MVO(mu, Q, robust=False, T=None, alpha=None, llambda=None, card=False, L=0.3, U=1, K=10):
    """
    #----------------------------------------------------------------------
    Use this function to construct an example of a MVO portfolio.
    #
    # An example of an MVO implementation is given below. You can use this
    # version of MVO if you like, but feel free to modify this code as much
    # as you need to. You can also change the inputs and outputs to suit
    # your needs.

    # You may use quadprog, Gurobi, or any other optimizer you are familiar
    # with. Just be sure to include comments in your code.

    # *************** WRITE YOUR CODE HERE ***************
    #----------------------------------------------------------------------
    
    Returns portfolio weights based on MVO.

    Inputs:
        mu (np.ndarray): n x 1 vector of expected asset returns
        Q (np.ndarray): n x n matrix of asset covariances
        robust (bool): flag for selecting robust MVO
        T (int): number of observations
        alpha (float): alpha value for ellipsoidal robust MVO
        llambda (float): lambda value for ellipsoidal robust MVO
    
    Returns:
        x (np.ndarray): n x 1 vector of estimated asset weights for the market portfolio

    """

    # Find the total number of assets
    n = len(mu)

    # Set the target as the average expected return of all assets
    targetRet = np.mean(mu)

    # Disallow short sales
    lb = np.zeros(n)

    # Add the expected return constraint
    A = -1 * mu.T
    b = -1 * targetRet

    # Constrain weights to sum to 1
    Aeq = np.ones([1, n])
    beq = 1

    # Define and solve using CVXPY
    x = cp.Variable(n)

    if not robust and not card:
        prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(x, Q)),
                        [A @ x <= b,
                        Aeq @ x == beq,
                        x >= lb])
        
    elif robust and not card:
        # Calculate theta and epsilon for ellipsoidal robust MVO
        theta = np.sqrt((1/T) * np.multiply(np.diag(Q), np.eye(n)))
        epsilon = np.sqrt(chi2.ppf(alpha, n))
        
        prob = cp.Problem(cp.Minimize(((1 / 2) * cp.quad_form(x, Q)) + (llambda * A @ x) + (epsilon * cp.norm(theta @ x, p=2))),
                        [Aeq @ x == beq,
                        x >= lb])
    
    elif not robust and card:
        y = cp.Variable(n, boolean = True)
        prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(x, Q)),
                        [A @ x <= b,
                        Aeq @ x == beq,
                        x >= lb,
                        (cp.sum(y) <= K),
                        (x >= L*y), 
                        (x <= U*y)])
    
    elif robust and card:
        y = cp.Variable(n, boolean = True)
        # Calculate theta and epsilon for ellipsoidal robust MVO
        theta = np.sqrt((1/T) * np.multiply(np.diag(Q), np.eye(n)))
        epsilon = np.sqrt(chi2.ppf(alpha, n))
        
        prob = cp.Problem(cp.Minimize(((1 / 2) * cp.quad_form(x, Q)) + (llambda * A @ x) + (epsilon * cp.norm(theta @ x, p=2))),
                        [Aeq @ x == beq,
                        x >= lb,
                        (cp.sum(y) <= K),
                        (x >= L*y), 
                        (x <= U*y)])

    prob.solve(verbose=False, solver=cp.GUROBI)
    return x.value

def market_cap(r_mkt, R):
    '''
    Returns estimated market portfolio weights.

    Inputs:
        r_mkt (np.ndarray): T x 1 vector of market returns
        R (np.ndarray): T x n matrix of asset returns
    
    Returns:
        x (np.ndarray): n x 1 vector of estimated asset weights for the market portfolio
    '''
    T, n = R.shape

    # Define and solve using CVXPY
    x = cp.Variable(n)

    # Constrain weights to sum to 1
    Aeq = np.ones([1, n])
    beq = 1

    # Disallow short sales
    lb = np.zeros(n)

    # Define objective function
    error = cp.norm(r_mkt - (R @ x), p=2)
    objective = cp.Minimize(error)

    # Define constraints
    constraints = [Aeq @ x == beq,
                   x >= lb]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False, solver=cp.ECOS)

    return x.value


#edit so you can construct MVO 
def Sharpe_MVO(mu, Q):
        # Find the total number of assets
    n = len(mu)

    # Define and solve using CVXPY
    y = cp.Variable(n)
    k = cp.Variable()
    

    prob = cp.Problem(cp.Minimize(cp.quad_form(y, Q)), 
                      [np.transpose(mu)@y == 1,
                       np.transpose(np.ones(n))@y == k,
                       k >= 0,
                       y >= 0])

    
    # change verbose to True to output optimization
    # info to console

    prob.solve(verbose=False, solver=cp.ECOS)
    x = y.value/k.value

    return x


def Robust_Sharpe_Eps_MVO(mu, Q, T): #NOT USING
           # Find the total number of assets
    n = len(mu)

    # Set the target as the average expected return of all assets
    targetRet = np.mean(mu)

    # Define and solve using CVXPY
    y = cp.Variable(n)
    eps = 1.96
    theta = np.sqrt(np.diag(Q)/T)
    k = np.transpose(np.ones(n))@y 

    prob = cp.Problem(cp.Minimize(cp.quad_form(y, Q)), 
                      [(mu.T)@y - eps*cp.norm((theta)@y) >= 1, #form of problem in objective
                       cp.sum(y) >= 0,
                       y >= 0])
    # change verbose to True to output optimization
    # info to console

    prob.solve(verbose=False, solver=cp.ECOS)
    x = y.value/k.value

    return x


def Max_Sharpe_Min_Turn(mu, Q, x0, llambda=1):
    #towards minimizing turnover
        # Find the total number of assets
    n = len(mu)

    # Define and solve using CVXPY
    y = cp.Variable(n)
    k = cp.Variable()
    z = cp.Variable(n)
     #scalar big just turn over, and small is sharpe 
    
    prob = cp.Problem(cp.Minimize(cp.quad_form(y, Q)+(llambda*cp.sum(z))), 
                      [np.transpose(mu)@y == 1,
                       np.transpose(np.ones(n))@y == k,
                       z >= y - (k*x0),
                       z >= (k*x0) -y,
                       k >= 0,
                       y >= 0])
    
    prob.solve(verbose=False, solver=cp.ECOS)
    x = y.value/k.value

    return x