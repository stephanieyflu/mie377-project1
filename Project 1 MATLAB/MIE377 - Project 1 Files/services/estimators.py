import numpy as np
import pandas as pd
import cvxpy as cp

def OLS(returns, factRet):
    # Use this function to perform a basic OLS regression with all factors.
    # You can modify this function (inputs, outputs and code) as much as
    # you need to.

    # *************** WRITE YOUR CODE HERE ***************
    # ----------------------------------------------------------------------

    # Number of observations and factors
    [T, p] = factRet.shape

    # Data matrix
    X = np.concatenate([np.ones([T, 1]), factRet.values], axis=1)

    # Regression coefficients
    B = np.linalg.solve(X.T @ X, X.T @ returns)

    # Separate B into alpha and betas
    a = B[0, :]
    V = B[1:, :]

    # Residual variance
    ep = returns - X @ B
    sigma_ep = 1 / (T - p - 1) * np.sum(ep.pow(2), axis=0)
    D = np.diag(sigma_ep)

    # Factor expected returns and covariance matrix
    f_bar = np.expand_dims(factRet.mean(axis=0).values, 1)
    F = factRet.cov().values

    # Calculate the asset expected returns and covariance matrix
    mu = np.expand_dims(a, axis=1) + V.T @ f_bar
    Q = V.T @ F @ V + D

    # Sometimes quadprog shows a warning if the covariance matrix is not
    # perfectly symmetric.
    Q = (Q + Q.T) / 2

    return mu, Q

def Lasso(returns, factRet, S=0.001):

    #S = sum of regression coefficients
    #lamb = 0.01
    #S = 0.0001

    # Number of observations and factors
    [T, p] = factRet.shape #T x p (36,8)
    [T, n] = returns.shape #T x n (36,20)

    B = np.zeros((p+1, n)) #B matrix of p x n (9,20)

    # Data matrix
    X = np.concatenate([np.ones([T, 1]), factRet.values], axis=1)
    

    #use for loop to go through each asset
    for i in range(n):

        # Regression coefficients
        b = cp.Variable(p+1) #make a vector of size p
        y = cp.Variable(p)    
        r = returns.iloc[:,i].values 

        n = cp.norm(b[1:], p=1) #l1 norm

        obj = cp.Minimize(cp.sum_squares((r - X@b)))
        constraints = [y >= -1*b[1:], y >= b[1:], (cp.sum(y) <= S), b[0] == 0] #set constraints

        prob = cp.Problem(obj, constraints)
        prob.solve(verbose=False, solver=cp.ECOS)

        B[:,i] = b.value

    a = B[0, :]
    V = B[1:, :]

    # Residual variance
    ep = returns - X @ B
    sigma_ep = 1 / (T - p - 1) * np.sum(ep.pow(2), axis=0)
    D = np.diag(sigma_ep)

    # Factor expected returns and covariance matrix
    f_bar = np.expand_dims(factRet.mean(axis=0).values, 1)
    F = factRet.cov().values

    # Calculate the asset expected returns and covariance matrix
    mu = np.expand_dims(a, axis=1) + V.T @ f_bar
    Q = V.T @ F @ V + D

    # Sometimes quadprog shows a warning if the covariance matrix is not
    # perfectly symmetric.
    Q = (Q + Q.T) / 2

    return mu, Q
  
def PCA(returns, p=3):
    '''
    Returns mu and Q estimates based on PCA.

    Inputs:
        returns (pd.DataFrame): T x n matrix of asset returns
        p (int): number of PCs to extract

    Returns:
        mu (np.ndarray): n x 1 vector of expected asset returns
        Q (np.ndarray): n x n matrix of asset covariances
    '''
    [T, n] = returns.shape

    ### PCA ###

    I = np.ones([T, 1])
    r_bar = (1/T) * ((returns.values).T @ I)

    # Centre the returns
    R_bar = returns.values - I @ (r_bar.T)

    # Estimate the biased covariance matrix
    Q_biased = (1/T) * (R_bar.T @ R_bar)

    # Perform eigenvalue decomposition
    w, v = np.linalg.eig(Q_biased) # w = eigenvalues, v = eigenvectors

    # Construct matrix of PCs
    P = R_bar @ v

    # Choose top p PCs
    P1 = P[:, :p]
    factRet = pd.DataFrame(P1)

    ### OLS ###

    # Number of observations and factors
    [T, p] = factRet.shape

    # Data matrix
    X = np.concatenate([np.ones([T, 1]), factRet], axis=1)

    # Regression coefficients
    B = np.linalg.solve(X.T @ X, X.T @ returns)

    # Separate B into alpha and betas
    a = B[0, :]
    V = B[1:, :]

    # Residual variance
    ep = returns - X @ B
    sigma_ep = 1 / (T - p - 1) * np.sum(ep.pow(2), axis=0)
    D = np.diag(sigma_ep)

    # Factor expected returns and covariance matrix
    f_bar = np.expand_dims(factRet.mean(axis=0), 1)
    F = factRet.cov().values

    # Calculate the asset expected returns and covariance matrix
    mu = np.expand_dims(a, axis=1) + V.T @ f_bar
    Q = V.T @ F @ V + D

    # Sometimes quadprog shows a warning if the covariance matrix is not
    # perfectly symmetric.
    Q = (Q + Q.T) / 2

    return mu, Q
