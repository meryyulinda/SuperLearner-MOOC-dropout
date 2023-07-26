import numpy as np
from scipy import optimize
import pandas as pd

# trim first, make sure there is no 0.0 or 1.0 in predicted outcomes of base learner.
# and then, change it to logit = log(x / 1-x).
# this transform BL predict that ranges [0,1] to logit (-~,~). so that when combined, creating a more linear relationship

def trimLogit(x, trim=0.00001):
    if x < trim:
        x = np.where(x<trim, trim, x)
    elif x > trim:
        x = np.where(x>(1-trim), (1-trim), x)
    trimmedLogit = np.log(x/(1-x))
    return trimmedLogit

# *   plogis = change it to logistic cumulative distribution function (CDF). This is simply the logistic function used in LogReg
# *   dlogis = change it to logistic density

def Rf_log1pexp(x):
    if(x <= float(18)):
        return np.log1p(np.exp(x))
    if(x > float(33.3)):
        return x
    # else: 18.0 < x <= 33.3:
    return x + np.exp(-x)

def plogis(x, location=0, scale=1, lower_tail=True, log_p=False):
  # if(x==np.nan or location==np.nan or scale==np.nan):
  #   return x + location + scale
    
    if(scale <= 0):
        return np.nan
  
    if(x==np.nan):
        return np.nan 

    x = (x-location)/scale  
  
    def R_P_bounds_Inf_01(x):
        if(x < 0):
          # return R_DT_0
          if(lower_tail):
            return -np.inf if(log_p) else 0
          else:
            return 0 if(log_p) else 1
    
        elif(x > 0):
          # return R_DT_1
          if(lower_tail):
            return 0 if(log_p) else 1
          else:
            return -np.inf if(log_p) else 0
    
    R_P_bounds_Inf_01(x)
  
    if(log_p):
        return -Rf_log1pexp(np.where(lower_tail, -x , x)) # log(1 / (1+exp(+-x))) = - log(1+exp(+-x))
    elif(log_p==False):
        return 1/(1+np.exp(np.where(lower_tail, -x , x)))

def dlogis(x, location=0, scale=1, give_log=False):
#ifdef IEEE_754
    # if (np.isnan(x) or np.isnan(location) or np.isnan(scale)):
    #   return x + location + scale;
#endif
    if (scale <= 0.0):
        return float("nan")

    x = np.abs((x - location) / scale);
    e = np.exp(-x);
    f = 1.0 + e;

    if give_log==True:
        return -(x + np.log(scale * f * f))
    if give_log==False:
        return e / (scale * f * f)

##### NNloglik method

def NNloglik(x, y, start_value):
    x = x
    y = np.array(y)
    start_value = start_value

    def fmin(beta, x, y):
        p = np.matmul(x, beta).apply(plogis)  # this is LogReg 1/(1+exp(X*beta))
        p = np.vstack((np.array(p),np.array(y)))
        p[0,:] = np.where(p[1,:]==1, np.log(p[0,:]), np.log(1-p[0,:])) # getting the [y=1: log(p); y=0: log(1-p)] for calculating sum of log likelihood later on
        p = p[0,:]
        return -np.sum(2 * p) # calculate binomial residual deviance = 2(LLsaturated-LLproposed) = 2(0 - LLproposed) = -2(log (odds)). smaller is better, this is that we'd like to optimize.

    # gmin is derivative of fmin
    def gmin(beta, x, y):
        eta = np.matmul(x,beta)
        p = eta.apply(plogis)
        p = np.vstack((np.array(p),np.array(y))) 
        p[0,:] = np.where(p[1,:]==1, eta.apply(dlogis)*1/p[0,:], eta.apply(dlogis)*-1/(1-p[0,:])) # getting the [y=1: 1/p; y=0: -1/(1-p)] 
        p = p[0,:]
        return -2 * np.matmul(p.T, x)

    # optimization begins here
    fit = optimize.minimize(x0 = start_value,
                            args=(x, y),
                            fun=fmin,
                            jac=gmin,
                            bounds=[(0,1)]*len(start_value), # bound for each beta coeff. to be in range [0,1]
                            method='L-BFGS-B'
                            )
    
    beta_metalearner = np.array(fit.x/sum(fit.x)) #normalize coefficients so that it summed up to 1.00
    if np.isnan(beta_metalearner).any() == True:
        beta_metalearner = np.array([1]*len(beta_metalearner))
        beta_metalearner = beta_metalearner/sum(beta_metalearner)

    [print(model_name, coeff) for model_name, coeff in zip(['SVM','LogReg','KNN','NB','RF','XGB'], beta_metalearner)]
    print('\n===============================================\n')

    return beta_metalearner

# REFERENCES    
#video reference 1: https://www.youtube.com/watch?v=nxtH8oTLrGI&list=PLLTSM0eKjC2cYVUoex9WZpTEYyvw5buRc&index=6 (logreg: likelihood ratio test)
#video reference 2: https://www.youtube.com/watch?v=ARfXDSkQf1Y (odds & log odds)
#video reference 3: https://www.youtube.com/watch?v=J0yuLu3oLuU (logreg: likelihood & deviance)
#video reference 4: https://www.youtube.com/watch?v=bhTIpGtWtzQ (MLE vs Least Squares)
#plogis reference: https://github.com/SurajGupta/r-source/blob/master/src/nmath/plogis.c and https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/Logistic and https://kaskr.github.io/adcomp/dpq_8h_source.html 
#switching between probability & log odds reference: https://www.montana.edu/rotella/documents/502/Lecture_03_R_code.pdf 
#Logistic Regression technique used by sklearn: minimization negative log likelihood using regularization https://stackoverflow.com/questions/24935415/logistic-regression-function-on-sklearn 
# https://machinelearningmastery.com/bfgs-optimization-in-python/
# https://empirical-methods.com/logistic-regression-and-friends.html
# https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/Logistic (the logistic distribution)
# https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html#optimize-minimize-slsqp
# https://docs.scipy.org/doc/scipy/tutorial/optimize.html 