import pandas as pd
import numpy as np
from scipy.stats import f as fdist
from scipy.stats import t as tdist
from scipy.stats import chi2 as chisq
from sklearn.base import BaseEstimator, ClassifierMixin
inv = np.linalg.inv

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
        
        # Estimate the variance of the error terms
        u_var_hat = rss/n-k
        
        # Estimate the variance of beta_hat
        beta_hat_var = u_var_hat*A
        
        # Calculate tss, ess and rss
        tss = y.var()*n
        ess = tss - rss
        
        # Calculate R-squared and Adjusted R-squared
        rsq = ess/tss
        rsq_adj = 1 - (rss/(n-k))/(tss/(n-1))
        
        # Estimated Variance of y_hat
        y_hat_var = np.array([X[i, :]@beta_hat_var@X[i, :].T for i in range(X.shape[0])]).reshape(-1, 1)
        
        # Estimated Variance of u_hat
        u_hat_var = y_hat_var + u_var_hat
        
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
        self.u_var_hat = u_var_hat
        self.rsq = rsq
        self.rsq_adj = rsq_adj
        self.aic = rss/n + 2*k/n
        self.bic = rss/n + 2*k*np.log(n)/n
        self.y_hat_var = y_hat_var
        self.u_hat_var = u_hat_var

        return self
    
    
    
        
    def predict(self, X):
        if self.fit_intercept:
            X = self._remake_X(X)
            
        return (X@self.beta_hat)
    
    def prediction_interval(self, X, conf=0.95):
        res = {}
        res['est'] = self.predict(X).ravel()
        if self.fit_intercept:
            X = self._remake_X(X)
            
        y_hat_var = np.array([X[i:(i+1), :]@self.beta_hat_var@X[i, :].T 
                              for i in range(X.shape[0])]).reshape(-1, 1)
        
        u_hat_var = y_hat_var + self.u_var_hat
        conf_width = (tdist.ppf((1+conf)/2, self.n - self.k)*u_hat_var).ravel()
        res['low'] = res['est'] - conf_width
        res['high'] = res['est'] + conf_width
        
        return res
        
    def confidence_interval(self, X, conf=0.95):
        res = {}
        res['est'] = self.predict(X).ravel()

        if self.fit_intercept:
            X = self._remake_X(X)
            
        y_hat_var = np.array([X[i:(i+1), :]@self.beta_hat_var@X[i, :].T 
                              for i in range(X.shape[0])]).reshape(-1, 1)
        
        conf_width = (tdist.ppf((1+conf)/2, self.n - self.k)*y_hat_var).ravel()
        res['low'] = res['est'] - conf_width
        res['high'] = res['est'] + conf_width
        
        return res
    
    
        
        
    
    def score(self, X, y):
        lm = LinearReg(self.fit_intercept)
        lm.fit(X, y)
        return lm.rsq_adj




    