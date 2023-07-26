from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import cudf as cf
import cupy as cp
import pandas as pd
import numpy as np
import time, datetime

def hyperparameter_tuning(tuning_mode, 
                          cpu_or_gpu,
                          model_name, estimator, hyperparameter_dict,  
                          x_train, y_train,
                          f1scoring, cv):
    
    start_tuning = datetime.datetime.now()
    print(f'Hyperparameter tuning for {model_name}...({cpu_or_gpu})')
    
    if model_name in ['NB', 'Naive Bayes'] and cpu_or_gpu == 'GPU':
        x_train = x_train.to_numpy()
    
    if tuning_mode == 'grid':
        estimator = GridSearchCV(estimator = estimator,
                                    param_grid = hyperparameter_dict,
                                    scoring = f'f1_{f1scoring}',
                                    cv = cv)
    elif tuning_mode == 'random':
        estimator = RandomizedSearchCV(estimator = estimator,
                                    param_distributions = hyperparameter_dict,
                                    scoring = f'f1_{f1scoring}',
                                    cv = cv)
        
    estimator.fit(x_train, y_train)

    hyperparamtuned_best_score = (pd.DataFrame(estimator.cv_results_)
                                  .sort_values('rank_test_score')
                                  .head(1)[['mean_test_score']]
                                  .values)
    print(f'\nBest hyperparameter for {model_name}:\n{estimator.best_params_}\n')
    print(f'Train score:\nF1-score ({f1scoring}) = {hyperparamtuned_best_score[0][0]}\n')
    print(f'--- {model_name} ({cpu_or_gpu}) IS TUNED IN {datetime.datetime.now()-start_tuning} ---')
    print('====================================================\n\n')

    return estimator