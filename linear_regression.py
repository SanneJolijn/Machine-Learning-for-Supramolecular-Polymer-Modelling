#%%
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_squared_error,
                             r2_score)
from sklearn.model_selection import (train_test_split,
                                     cross_val_score,
                                     RandomizedSearchCV,
                                     KFold)
import matplotlib.pyplot as plt
import csv
from numpy import (mean,
                   std)

import pandas as pd

import seaborn as sns
from sklearn.datasets import make_classification, make_circles
from sklearn import preprocessing
from sklearn.svm import SVR
import csv
from utils import import_xlsx_data

#%%
def linear_regression(n_outer, random_state, X, y):
    outer_cv = KFold(n_splits=n_outer, shuffle=True, random_state=random_state)
    outer_results = list()

    df_metrics = pd.DataFrame()

    for train_ix, test_ix in outer_cv.split(X):
        # split data
        X_train, X_test = X.loc[train_ix], X.loc[test_ix]
        y_train, y_test = y.loc[train_ix], y.loc[test_ix]

        model = LinearRegression().fit(X_train, y_train)

        itc = model.intercept_
        coef = model.coef_

        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        outer_results.append(r2)

        metrics = [r2, rmse, itc, coef] 
        df_metrics = pd.concat([df_metrics, pd.DataFrame(data=[metrics])])

    print('R squared: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))

    return df_metrics


#%%
#Determine all variables
np.random.seed(12)
random_state = 12
n_outer = 10


# %%
#Import our data
data = import_xlsx_data('upy AA input.xlsx')
df = data.drop(data.columns[0], axis=1)
X, y = df.drop('Turbidity', axis=1), df[['Turbidity']]

#%%
#Perform the linear regression on all compounds and turbidity
df_performance_LR = linear_regression(n_outer, random_state, X, y)
df_performance_LR.to_csv("regression performance.csv") 

#%%
#Perform the linear regression on c4 and turbidity
X1, y1 = df[['c4']], df[['Turbidity']]
df_performance_c4 = linear_regression(n_outer, random_state, X1, y1)
df_performance_c4.to_csv("regression c4 performance.csv") 

# %%
#Ablation studies
data2 = import_xlsx_data('upy AA input 3.xlsx')
df = data2.drop(data2.columns[0], axis=1)
X2, y2 = df.drop('Turbidity', axis=1), df[['Turbidity']]

#%%
df_performance_LR2 = linear_regression(n_outer, random_state, X2, y2)
df_performance_LR2.to_csv("regression ablation.csv") 
