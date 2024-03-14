import cvxpy as cp
import numpy as np


def MVO(mu, Q):
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

    # constrain weights to sum to 1
    Aeq = np.ones([1, n])
    beq = 1

    # Define and solve using CVXPY
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(x, Q)),
                      [A @ x <= b,
                       Aeq @ x == beq,
                       x >= lb])
    # change verbose to True to output optimization
    # info to console

    prob.solve(verbose=False, solver=cp.ECOS)
    return x.value

def market_cap(r_mkt, R):
    '''
    Return estimated market portfolio weights
    '''
    T, n = R.shape
    x = cp.Variable(n)

    Aeq = np.ones([1, n])
    beq = 1
    lb = np.zeros(n)

    error = cp.norm(r_mkt - (R @ x), p=2)
    objective = cp.Minimize(error)
    constraints = [x >= lb, 
                   Aeq @ x == beq]
    
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return x.value