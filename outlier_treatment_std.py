import cudf as cf
import cupy as cp
import pandas as pd
import numpy as np

def outlier_treatment_std(df, cols, threshold):
    
    if 'course_category' in cols: # ignore "course_category" columns
        pass
    
    else:
        upper_bound = df[cols].mean() + threshold*df[cols].std()
        lower_bound = df[cols].mean() - threshold*df[cols].std()
        print(f'\n{cols}: upper bound = {upper_bound} | lower bound = {lower_bound}')
        
        extreme_small_values = df[ df[cols]<lower_bound ]
        extreme_big_values = df[ df[cols]>upper_bound ]

        if lower_bound < 0 and not cols.endswith('_weeklydiff'): #ensure there is no values replaced by negative lower bound, except '_weeklydiff' cols
            print(len(extreme_big_values), f'values in {cols} are bigger than upper bound')
            if len(extreme_big_values) < 1e4: #only treat the outliers if len(outliers) < 10,000
                df.loc[ df[cols]>upper_bound, cols ] = upper_bound
                print('Extremely big values already changed with upper bound!\n')
            return df[cols]

        else:
            print(len(extreme_big_values), f'values in {cols} are bigger than upper bound')
            print(len(extreme_small_values), f'values in {cols} are smaller than lower bound')
            if len(extreme_small_values) < 1e4:
                df.loc[ df[cols]<lower_bound, cols ] = lower_bound 
                print('Extremely small values already changed with lower bound!')
            if len(extreme_big_values) < 1e4:
                df.loc[ df[cols]>upper_bound, cols ] = upper_bound
                print('Extremely big values already changed with upper bound!')
            return df[cols]