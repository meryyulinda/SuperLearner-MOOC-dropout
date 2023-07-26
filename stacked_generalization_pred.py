import time, datetime
import numpy as np
import pandas as pd
import cupy as cp
import cudf as cf
from sklearn.metrics import f1_score, classification_report

def stacked_generalization_pred(list_metalearner_name=None, list_metalearner_estimator=None, 
                                f1scoring='macro',
                                Z_train_crossval=None,
                                Z_train=None, Z_test=None,
                                cpu_or_gpu=None):

    start_time = datetime.datetime.now()
    print(f'({cpu_or_gpu}) STACKED GENERALIZATION RESULTS ...')
    print(f'* f1scoring={f1scoring}')
    report_f1score = []
    threshtuned = 0

    for meta_name, meta_estimator in zip(list_metalearner_name, list_metalearner_estimator):

        report_f1score.append(['Stacked Generalization', meta_name, cpu_or_gpu, f1scoring])
        meta_estimator.fit(Z_train_crossval.to_cupy().get(), Z_train['dropout'].to_cupy().get().ravel())

        for Z, which_dataset in zip([Z_train, Z_test], ['Train','Test']):

            y = Z['dropout'].to_cupy().get().ravel()
            Z['SG_'+meta_name+'_predprob'] = meta_estimator.predict_proba(Z.drop(columns='dropout').iloc[:,:6].to_cupy().get())[:,1]

            # Threshold tuning
            thresholds = 0.5
            Z['SG_'+meta_name+'_binary'] = cp.where(Z['SG_'+meta_name+'_predprob']>thresholds, 1, 0)
            f1scores = f1_score(Z['dropout'].to_cupy().get(), 
                                Z['SG_'+meta_name+'_binary'].to_cupy().get(), 
                                average=f1scoring)
            
            print(f'{which_dataset} score of SG-{meta_name}:')
            print('F1-Score (%s) = %.5f \n' % (f1scoring, f1scores))
            classif_report = classification_report(Z['dropout'].to_cupy().get(), 
                                                   Z['SG_'+meta_name+'_binary'].to_cupy().get(), 
                                                   digits=5, 
                                                   output_dict=True)
            f1scores_class1 = classif_report['1']['f1-score']
            f1scores_class0 = classif_report['0']['f1-score']
            report_f1score[-1].extend([f1scores, f1scores_class1, f1scores_class0])
          
    Z_train.to_csv(f'SG_train_{cpu_or_gpu}.csv')
    Z_test.to_csv(f'SG_test_{cpu_or_gpu}.csv')
    
    print('BEST METALEARNER for STACKED GENERALIZATION:')
    list_f1score = [report_f1score[i][7] for i in range(len(report_f1score))]
    index_bestSGmeta = np.argmax(list_f1score)
    name_bestSGmeta = report_f1score[index_bestSGmeta][1]
    print(name_bestSGmeta,'\n')
    print('Train score report:')
    print(classification_report(Z_train['dropout'].to_cupy().get(), 
                                Z_train['SG_'+name_bestSGmeta+'_binary'].to_cupy().get(), 
                                digits=5))
    print('\nTest score report:')
    print(classification_report(Z_test['dropout'].to_cupy().get(), 
                                Z_test['SG_'+name_bestSGmeta+'_binary'].to_cupy().get(), 
                                digits=5))
    
    print('---TOTAL EXECUTION TIME: %s---' % (datetime.datetime.now()-start_time))
    return report_f1score