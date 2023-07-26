import numpy as np
from scipy import optimize
from sklearn.metrics import roc_auc_score
import pandas as pd

def auc_maximization(Z, y, beta_start_value):
    
    def crossprod_auc(beta, Z, y):
        crossprod = np.matmul(Z, beta) # this is LinReg (X * beta)
        rank_loss = 1 - roc_auc_score(y, crossprod) # getting the rank loss (1-AUC)
        return rank_loss # this is that we want to optimize
    
    # optimization starts here
    fit = optimize.minimize(fun=crossprod_auc,
                            x0 = beta_start_value,
                            args=(Z, y),
                            bounds=[(0,1)]*len(beta_start_value), # bound for each beta coeff. to be in range [0,1]
                            method='L-BFGS-B'
                           )

    beta_metalearner = np.array(fit.x/sum(fit.x)) #normalize coefficients so that it summed up to 1.00
    if np.isnan(beta_metalearner).any() == True:
        beta_metalearner = np.array([1]*len(beta_metalearner))
        beta_metalearner = beta_metalearner/sum(beta_metalearner)

    [print(model_name, coeff) for model_name, coeff in zip(['SVM','LogReg','KNN','NB','RF','XGB'], beta_metalearner)]
    print('\n===============================================\n')

    return beta_metalearner