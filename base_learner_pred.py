import datetime, time
import numpy as np
import pandas as pd
import cupy as cp
import cudf as cf
from sklearn.metrics import f1_score, classification_report

def base_learner_pred(list_baselearner_name=None,
                      f1scoring='macro',
                      Z_train=None, Z_test=None,
                      cpu_or_gpu=None):

    start_time = datetime.datetime.now()
    print(f'({cpu_or_gpu}) BASE LEARNER RESULTS ...')
    print(f'* f1scoring={f1scoring}')
    report_f1score = []
    

    for base_learner in list_baselearner_name:

        report_f1score.append([base_learner, '', cpu_or_gpu, f1scoring])

        for Z, which_dataset in zip([Z_train, Z_test], ['Train','Test']):

            y = Z['dropout'].to_cupy().get().ravel()
  
            # Threshold tuning
            thresholds = 0.5
            Z[base_learner+'_binary'] = cp.where(Z[base_learner]>thresholds, 1, 0)
            f1scores = f1_score(Z['dropout'].to_cupy().get(), 
                                Z[base_learner+'_binary'].to_cupy().get(), 
                                average = f1scoring)

            print(f'{which_dataset} score of {base_learner}:')
            print('F1-Score (%s) = %.5f \n' % (f1scoring, f1scores))
            classif_report = classification_report(Z['dropout'].to_cupy().get(), 
                                                   Z[base_learner+'_binary'].to_cupy().get(), 
                                                   digits=5, 
                                                   output_dict=True)
            f1scores_class1 = classif_report['1']['f1-score']
            f1scores_class0 = classif_report['0']['f1-score']
            report_f1score[-1].extend([f1scores, f1scores_class1, f1scores_class0])

    print('BEST BASE LEARNER:')
    list_f1score = [report_f1score[i][7] for i in range(len(report_f1score))]
    index_bestBL = np.argmax(list_f1score)
    name_bestBL = report_f1score[index_bestBL][0]
    print(name_bestBL,'\n')
    print('Train score report:')
    print(classification_report(Z_train['dropout'].to_cupy().get(), 
                                Z_train[name_bestBL+'_binary'].to_cupy().get(), 
                                digits=5))
    print('\nTest score report:')
    print(classification_report(Z_test['dropout'].to_cupy().get(), 
                                Z_test[name_bestBL+'_binary'].to_cupy().get(), 
                                digits=5))
    
    print('---TOTAL EXECUTION TIME: %s---' % (datetime.datetime.now()-start_time))
    return report_f1score