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

    prob.solve(verbose=True, solver=cp.ECOS)
    x = y.value/k.value

    return x

def Max_Sharpe_Min_Turn_1(mu, Q, x0, llambda=2):
    #towards minimizing turnover
        # Find the total number of assets
    n = len(mu)

    # Define and solve using CVXPY
    y = cp.Variable(n)
    k = cp.Variable()
    z = cp.Variable(n)
     #scalar big just turn over, and small is sharpe 
    
    print(x0.shape)
    print(k)
    prob = cp.Problem(cp.Minimize(cp.quad_form(y, Q)+(llambda*cp.sum(z))), 
                      [np.transpose(mu)@y == 1,
                       np.transpose(np.ones(n))@y == k,
                       z >= y - (k*x0),
                       z >= (k*x0) -y,
                       k >= 0,
                       y >= 0])
    
    prob.solve(verbose=True, solver=cp.ECOS)
    x = y.value/k.value

    return x

def Max_Sharpe_Min_Turn_2(mu, Q, x0, llambda = 0.5):
    #towards maximizing sharpe
        # Find the total number of assets
    n = len(mu)

    # Define and solve using CVXPY
    y = cp.Variable(n)
    k = cp.Variable()
    z = cp.Variable(n)
     #scalar big just turn over, and small is sharpe 
    
    print(x0.shape)
    print(k)
    prob = cp.Problem(cp.Minimize(cp.quad_form(y, Q)+(llambda*cp.sum(z))), 
                      [np.transpose(mu)@y == 1,
                       np.transpose(np.ones(n))@y == k,
                       z >= y - (k*x0),
                       z >= (k*x0) -y,
                       k >= 0,
                       y >= 0])
    
    prob.solve(verbose=True, solver=cp.ECOS)
    x = y.value/k.value

    return x
    


