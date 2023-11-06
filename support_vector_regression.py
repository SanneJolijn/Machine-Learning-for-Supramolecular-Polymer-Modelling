# Import necessary packages
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

# Define a function for Support Vector Regression (SVR)
def support_vector_regression(random_state, n_outer, n_inner, X, y, parameter_grid, iterations):
    # Define the outer and inner cross-validation splits
    outer_cv = KFold(n_splits=n_outer, shuffle=True, random_state=random_state)
    inner_cv = KFold(n_splits=n_inner, shuffle=True, random_state=random_state)

    estimator = SVR() # Support Vector Regression as the estimator

    outer_results = list()
    pointer = 0
    df_randomsearch = pd.DataFrame()
    df_metrics = pd.DataFrame()

    for train_ix, test_ix in outer_cv.split(X):
        metrics = []
        # Split data into training and test sets
        X_train, X_test = X.loc[train_ix], X.loc[test_ix]
        y_train, y_test = y.loc[train_ix], y.loc[test_ix]

        # Define RandomizedSearchCV for hyperparameter tuning
        random_search = RandomizedSearchCV(estimator=estimator,
                                           param_distributions=parameter_grid,
                                           n_iter=iterations, cv=inner_cv,
                                           scoring='r2',
                                           n_jobs=-1, random_state=random_state)

        # Train the model using the training sets
        result = random_search.fit(X_train, y_train)
        
        parameter_results = pd.DataFrame.from_dict(result.cv_results_, orient='columns')

        df_randomsearch = pd.concat([df_randomsearch, parameter_results])

        best_model = result.best_estimator_

        itc = best_model.intercept_
        coef = best_model.coef_

        y_pred = best_model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        outer_results.append(r2)

        metrics = [r2, rmse, itc, coef] #result.best_params_
        df_metrics = pd.concat([df_metrics, pd.DataFrame(data=[metrics])])

        pointer += 1
            
    print('R squared: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))

    return df_randomsearch, df_metrics

# Define variables
np.random.seed(12)
random_state = 12

n_outer = 10
n_inner = 10

parameter_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['linear']
}  

iterations = 2000

# Import data using a custom utility function
data = import_xlsx_data('upy AA input.xlsx')
df = data.drop(data.columns[0], axis=1)
X, y = df.drop('Turbidity', axis=1), df[['Turbidity']]

# Perform Support Vector Regression on all compounds and turbidity
df_results_SVR, df_performance_SVR = support_vector_regression(random_state, n_outer, n_inner, X, y, parameter_grid, iterations)
df_performance_SVR.to_csv("regression SVR performance.csv")
df_results_SVR.to_csv("regression SVR results.csv")

# Perform Support Vector Regression on c4 and turbidity
X1, y1 = df[['c4']], df[['Turbidity']]
df_results_SVR_c4, df_performance_SVR_c4 = support_vector_regression(random_state, n_outer, n_inner, X1, y1, parameter_grid, iterations)
df_performance_SVR_c4.to_csv("regression c4 SVR performance.csv")
df_results_SVR_c4.to_csv("regression c4 SVR results.csv")

#Ablation studies
data2 = import_xlsx_data('upy AA input 3.xlsx')
df = data2.drop(data2.columns[0], axis=1)
X2, y2 = df.drop('Turbidity', axis=1), df[['Turbidity']]

# Perform Support Vector Regression for ablation studies
df_results_SVR2, df_performance_SVR2 = support_vector_regression(random_state, n_outer, n_inner, X2, y2, parameter_grid, iterations)
df_performance_SVR2.to_csv("regression SVR ablation.csv")
df_results_SVR2.to_csv("regression results SVR ablation.csv")

