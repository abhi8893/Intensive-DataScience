import numpy as np
from scipy.stats import f as fdist
from sklearn.base import BaseEstimator, ClassifierMixin


class LinearReg(BaseEstimator, ClassifierMixin):
    
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
    
    @staticmethod
    def _remake_X(X):
        if not np.all(X == 0):
            return np.hstack((np.ones((X.shape[0], 1)), X))
        else:
            return np.ones((X.shape[0], 1))

    @staticmethod
    def _remake_y(y):
        return y.reshape(-1, 1) # Check before reshaping?

    
    def fit(self, X, y):
        
        # Make proper X and y matrix
        if self.fit_intercept:
            if X is not None:
                X = self._remake_X(X)
            else:
                X = np.ones((y.shape[0], 1))
            
        y = self._remake_y(y)
        
        # Store n and k
        n, k = X.shape
        
        # Obtain estimate of beta =  beta_hat
        try:
            A = inv(X.T@X)
        except np.linalg.LinAlgError:
            A = np.zeros([X.shape[1]]*2)
            
        beta_hat = A@X.T@y
        
        # Obtain estimate of y = y_hat
        y_hat = X@beta_hat
        
        # Compute residual sum of squares
        u_hat = y - y_hat
        rss = (u_hat.T @ u_hat).ravel()[0]
        
        # Estimate the variance of beta_hat
        beta_hat_var = (rss/(n-k))*A
        
        # Calculate tss, ess and rss
        tss = y.var()*n
        ess = tss - rss
        
        # Calculate R-squared and Adjusted R-squared
        rsq = ess/tss
        rsq_adj = 1 - (rss/(n-k))/(tss/(n-1))
        
        # Perform F-test for overall significance
        if k != 1:
            dfn, dfd = k-1, n-k
            self.fstat = (ess/(dfn))/(rss/(dfd))
            self.ftest_pval =  1 - fdist.cdf(self.fstat, dfn, dfd)

        # store
        self.n = n
        self.k = k
        self.beta_hat = beta_hat
        self.rss = rss
        self.ess = ess
        self.tss = tss
        self.beta_hat_var = beta_hat_var
        self.u_hat = u_hat
        self.rsq = rsq
        self.rsq_adj = rsq_adj
        self.aic = rss/n + 2*k/n
        self.bic = rss/n + 2*k*np.log(n)/n

        return self
    
        
    def predict(self, X):
        X = self._remake_X(X)
        return X@self.beta_hat
    
    def predict_interval(self, X):
        raise NotImplementedError('This feature is not implemented yet!')
        
    
    def score(self, X, y):
        lm = LinearReg(self.fit_intercept)
        lm.fit(X, y)
        return lm.rsq_adj
    