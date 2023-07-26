import numpy as np
import pandas as pd
import datetime, time
import cudf as cf
import pandas as pd
import cupy as cp
import seaborn as sns
import matplotlib.pyplot as plt

def generate_Z(list_modelname_baselearner, list_estimators_baselearner,
               x, x_scaled, y, 
               which_dataset,
               cpu_or_gpu):
    
    print(f'({cpu_or_gpu}) Creating Z table for Predict Proba results on {which_dataset} set...\n')
    start_time = datetime.datetime.now()

    Z = cf.DataFrame({'dropout':y})

    for model_name, base_learner in zip(list_modelname_baselearner, list_estimators_baselearner):

        start_modelinfer = datetime.datetime.now()

        if model_name in ['SVM', 'KNN', 'LogReg']:
            Z = cf.concat([Z, cf.DataFrame({model_name : cp.array(base_learner.predict_proba(x_scaled))[:,1]})], axis=1)
#         elif model_name in ['NB'] and cpu_or_gpu == 'GPU':
#             Z = cf.concat([Z, cf.DataFrame({model_name : cp.array(base_learner.predict_proba(x.to_numpy()))[:,1]})], axis=1)
        else:
            Z = cf.concat([Z, cf.DataFrame({model_name : cp.array(base_learner.predict_proba(x))[:,1]})], axis=1)
        print(f'{model_name} INFERENCE TIME IS {datetime.datetime.now()-start_modelinfer}\n')
    
    print(f'---TOTAL INFERENCE TIME IS {datetime.datetime.now()-start_time}---\n')
    
    # Plot Correlation between Each Base Learner in Z table
    corr = Z.to_pandas().corr('spearman') # corr() using the Spearman method that could detect variable relationship in non-linearity settings
    labels_corr = np.where(np.abs(corr)>0.75,'S', # correlation more than +/- 75% labelled as 'STRONG CORRELATION'
                      np.where(np.abs(corr)>0.5,'M', # correlation more than +/- 50% labelled as 'MEDIUM CORRELATION'
                               np.where(np.abs(corr)>0.25,'W',''))) # correlation more than +/- 25% labelled as 'WEAK CORRELATION'
    sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)), square=True,
                center=0, annot=labels_corr, fmt='', linewidths=.5,
                cmap="vlag", cbar_kws={"shrink": 0.8})
    plt.title(f'Correlation between each Base Learner prediction on Z_{which_dataset} table')
    plt.show()
    
    return Z