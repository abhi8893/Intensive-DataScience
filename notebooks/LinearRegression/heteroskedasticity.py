from linear_regression import LinearReg
import numpy as np
from scipy.stats import f as fdist
from scipy.stats import t as tdist
from scipy.stats import chi2 as chisq

class IncompatibleLinearModelError(Exception):
    pass

def Ftest_LinearReg(lm1: LinearReg, lm2: LinearReg, sig):
    '''Perform the Ftest for two linear model fits of class LinearReg
    
    The restricted and unrestricted models are automatically assigned
    by accessing the k attribute of lm1 and lm2.
    
    Parameters
    ----------
    
    lm1, lm2: LinearReg
        fitted models of class LinearReg
        
    sig: int, optional (default=0.05)
        the significance level of the test
        
    Return -> True if the test is significant
    
    '''
    if (lm1.n == lm2.n) and (lm1.k != lm2.k): # NOTE: Python 3.8 assign n to a var 
        n = lm1.n
    else:
        raise(IncompatibleLinearModelError(
            'The linear models were fitted on datasets with varying observations')
             )
    
    lms = [lm1, lm2]
    r = np.argmin([lm.k for lm in lms])
    ur = [i for i in range(len(lms)) if i != r][0]
    
    lm_r, lm_ur = lms[r], lms[ur]
    
    rrss, urss = lm_r.rss, lm_ur.rss
    dfn = lm_ur.k - lm_r.k
    dfd = n - lm_ur.k
    
    F_stat = ((rrss - urss)/dfn)/(urss/dfd)
    critical_val = fdist.ppf(1 - sig, dfn, dfd)
    pval = 1 - fdist.cdf(F_stat, dfn, dfd)

    print('F statistic: {:.4f} with dfn: {}, and dfd: {}'.format(F_stat, dfn, dfd))
    print('Critical value: {:.4f}'.format(critical_val))
    print('p-value: {:.4f}'.format(pval))
    
    if F_stat > critical_val:
        return True
    
    return False


def breusch_pagan(X: np.array, y: np.array, sig):
    ''' Breush Pagan test for heteroskedasticity
    
    Parameters
    ----------
    X: np.array
        Regressor matrix
    
    y: np.array
        Regressand matrix
        
    '''
    lm = LinearReg().fit(X, y)

    # residual
    res = y - lm.predict(X)

    # residual squared
    res_sq = res**2

    # Null Hypothesis
    lm_H0 = LinearReg().fit(None, res_sq)

    # Alternate Hypothesis
    # Now regress the res_sq on X
    lm_H1 = LinearReg().fit(X, res_sq)
    
    return(Ftest_LinearReg(lm_H0, lm_H1, sig))