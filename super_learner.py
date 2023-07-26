import numpy as np
import cupy as cp
import datetime, time
import pandas as pd
import cudf as cf
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix, f1_score, log_loss, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def trimLogit(x, trim=0.00001):
    if x < trim:
        x = np.where(x<trim, trim, x)
    elif x > trim:
        x = np.where(x>(1-trim), (1-trim), x)
    trimmedLogit = np.log(x/(1-x))
    return trimmedLogit

# *   plogis = change it to logistic cumulative distribution function (CDF)
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


def super_learner(metalearner,  baselearner_weight,
                  f1scoring='macro',
                  Z_train=None, Z_test=None,
                  cpu_or_gpu=None):
    
    report_f1score = [['Super Learner', metalearner, cpu_or_gpu, f1scoring]]
    start_time = datetime.datetime.now()
    
    print(f'({cpu_or_gpu}) SUPER LEARNER RESULTS with \'{metalearner}\' metalearner ...')
    print(f'* f1scoring={f1scoring}')

    for Z, which_dataset in zip([Z_train, Z_test], ['Train', 'Test']):
        start_eachdata = datetime.datetime.now()

        # Combining Base Learners using the specified Metalearner & Optimized base learner weights
        Z = Z.copy()
        if metalearner == 'nnloglik':
            Z['combine_predprob'] = np.matmul(Z.drop(columns=['dropout']).to_pandas().applymap(trimLogit), 
                                              baselearner_weight).apply(plogis)
        elif metalearner == 'aucmaxim':
            Z['combine_predprob'] = np.matmul(Z.drop(columns=['dropout']).to_pandas(), 
                                              baselearner_weight)

        # Threshold tuning
        thresholds = 0.5
        Z['combine'] = cp.where(Z['combine_predprob']>thresholds, 1, 0)
        f1scores = f1_score(Z['dropout'].to_cupy().get(), 
                            Z['combine'].to_cupy().get(), 
                            average = f1scoring)
      
        Z.to_csv(f'SL{metalearner}_{which_dataset}_{cpu_or_gpu}.csv')
      
        # Score Reports
        print(f'{which_dataset} score:')
        print('Log Loss =', log_loss(Z['dropout'].to_cupy().get(), 
                                     Z['combine_predprob'].to_cupy().get()))
        print('AUC score =', roc_auc_score(Z['dropout'].to_cupy().get(),
                                           Z['combine'].to_cupy().get()))
        print('F1-Score (%s) = %.5f (with threshold=%.3f)\n' % (f1scoring, f1scores, thresholds))
        print(classification_report(Z['dropout'].to_cupy().get(), 
                                    Z['combine'].to_cupy().get(), 
                                    digits=5))
        classif_report = classification_report(Z['dropout'].to_cupy().get(), 
                                               Z['combine'].to_cupy().get(), 
                                               digits=5, 
                                               output_dict=True)
        f1scores_class1 = classif_report['1']['f1-score']
        f1scores_class0 = classif_report['0']['f1-score']
        report_f1score[-1].extend([f1scores, f1scores_class1, f1scores_class0])
        ConfusionMatrixDisplay(confusion_matrix(Z['dropout'].to_cupy().get(), 
                                                Z['combine'].to_cupy().get()),
                                display_labels=np.array(['Non-Dropout\n(0)','Dropout\n(1)'])
                              ).plot()
        plt.grid(False)
        plt.xlabel('\nPREDICTED', fontsize=15, style='italic', color='red')
        plt.ylabel('ACTUAL', fontsize=15, fontweight='bold', color='blue')
        plt.xticks(color='red')
        plt.yticks(color='blue')
        plt.show()

        # Plot Combine_predprob between class 1 and class 0
        plt.figure(figsize=(5,4))
        sns.kdeplot(data=Z.to_pandas(), x='combine_predprob', hue='dropout')
        plt.show()

        print(f'=={which_dataset} set done in %s==' % (datetime.datetime.now()-start_eachdata))
    
    print('---TOTAL EXECUTION TIME: %s---' % (datetime.datetime.now()-start_time))

    return report_f1score