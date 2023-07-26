from retry_decorator import *
from sklearn.model_selection import cross_val_predict
import cudf as cf
import cupy as cp
import time, datetime
from sklearn.metrics import log_loss, roc_auc_score, f1_score

#### >> FUNCTION: CROSS-VALIDATION ON BL FOR BUILDING META-LEARNER

@retry((Exception), tries=3)
def crossvalpredict(estimator, X, y, cv, n_jobs):
    start_crossval = datetime.datetime.now()
    return cross_val_predict(estimator=estimator, X=X, y=y, cv=cv, n_jobs=n_jobs, method='predict_proba')[:,1], (datetime.datetime.now()-start_crossval)
    
def train_baselearner_crossval(list_modelname, list_estimators, 
                               crossval, n_jobs, f1scoring,
                               x_train, x_train_scaled,
                               y_train,
                               cpu_or_gpu):
    
    print(f'=== CROSS-VALIDATION TRAINING TO BASE LEARNERS [ {cpu_or_gpu} ] ===\n')
    start_time = datetime.datetime.now()
    train_crossval_df = cf.DataFrame({'column to drop':[]})
    
    for model_name, estimator in zip(list_modelname, list_estimators):
        
        print(f'Cross-validation training for {model_name}...')
        
        if model_name in ['SVM', 'KNN', 'LogReg']:
            train_crossval, execution_time = crossvalpredict(estimator, x_train_scaled, y_train, crossval, n_jobs)
        
        else:
            train_crossval, execution_time = crossvalpredict(estimator, x_train, y_train, crossval, n_jobs)
        
        train_crossval_df = cf.concat([train_crossval_df, cf.DataFrame({model_name:train_crossval})], axis=1)
        print(f'---DONE IN {execution_time}---')
        print('Train score (Cross-validated risk):')
        print('Log Loss =', log_loss(y_train, train_crossval))
        print('AUC score =', roc_auc_score(y_train,  
                                           cp.where(cp.array(train_crossval)>0.5, 1, 0).get()))
        f1 = f1_score(y_train, 
                      cp.where(cp.array(train_crossval)>0.5, 1, 0).get(), 
                      average = f1scoring)
        print('F1-Score (%s) = %.5f\n\n' % (f1scoring, f1))
    
    Z_crossval_for_building_metalearner = train_crossval_df.drop(columns=['column to drop'])
 
    print('\n---TOTAL EXECUTION TIME: %s---\n' % (datetime.datetime.now()-start_time) )
    return Z_crossval_for_building_metalearner