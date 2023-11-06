#%%
import xgboost as xgb
from xgboost import (XGBClassifier,
                     XGBRegressor)

from sklearn.model_selection import (train_test_split,
                                     cross_val_score,
                                     RandomizedSearchCV,
                                     KFold)
from sklearn.metrics import (                            
                             confusion_matrix,
                             balanced_accuracy_score,
                             mean_squared_error,
                             mean_absolute_error,
                             r2_score)

import pandas as pd
import math
import csv

import numpy as np
from numpy import (mean,
                   std)

from utils import import_xlsx_data

# Define a function for XGBoost regression with hyperparameter tuning
def XGBoost_regression(random_state, n_outer, n_inner, X, y, parameter_grid, iterations):
    # Configure the cross-validation procedure
    outer_cv = KFold(n_splits=n_outer, shuffle=True, random_state=random_state)
    inner_cv = KFold(n_splits=n_inner, shuffle=True, random_state=random_state)

    # Define the model
    estimator = xgb.XGBRegressor()

    outer_results = list()
    pointer = 0
    df_randomsearch = pd.DataFrame()
    df_metrics = pd.DataFrame()    

    for train_ix, test_ix in outer_cv.split(X):
        metrics = []
        # Split data
        X_train, X_test = X.loc[train_ix], X.loc[test_ix]
        y_train, y_test = y.loc[train_ix], y.loc[test_ix]

        # Define random search
        random_search = RandomizedSearchCV(estimator=estimator, 
                                            param_distributions=parameter_grid, 
                                            n_iter=iterations, cv=inner_cv, 
                                            scoring='r2',
                                            n_jobs=-1, random_state=random_state)
            
        # Execute random search
        result = random_search.fit(X_train, y_train)
            
        parameter_results = pd.DataFrame.from_dict(result.cv_results_, orient='columns')
            
        df_randomsearch = pd.concat([df_randomsearch, parameter_results])

        # Get the best performing model fit on the whole training set
        best_model = result.best_estimator_

        # Evaluate model on the hold out dataset
        y_predicted = best_model.predict(X_test)

        # Evaluate the regression model
        r2 = r2_score(y_test, y_predicted)
        rmse = mean_squared_error(y_test, y_predicted, squared=False)
        ftr_importance = best_model.feature_importances_
        outer_results.append(r2)

        metrics = [r2, rmse, result.best_score_, ftr_importance] #result.best_params_
        df_metrics = pd.concat([df_metrics, pd.DataFrame(data=[metrics])])

        pointer += 1
            
        print(pointer)

    # Summarize the estimated performance of the model
    print('R squared: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))

    return df_randomsearch, df_metrics

# Set random seed and parameters
np.random.seed(12)
random_state = 12

n_outer = 10
n_inner = 10

parameter_grid = {
    'max_depth': [3, 4, 5, 6 ,7],
    'learning_rate': [0.05, 0.1, 0.15, 0.20],
    'subsample': [0.25, 0.5, 0.75, 1],
    'n_estimators': [10, 25, 50, 100, 150, 200, 250],
    'reg_alpha': [0, 0.1, 0.25],
    'colsample_by_tree': [0.5, 0.75, 1]
}

iterations = 2000

#Import data
data = import_xlsx_data('upy AA input.xlsx')
df = data.drop(data.columns[0], axis=1)
X, y = df.drop('Turbidity', axis=1), df[['Turbidity']]

#Perform XGBoost hyperparameter tuning
df_results, df_performance = XGBoost_regression(random_state, n_outer, n_inner, X, y, parameter_grid, iterations)
df_results.to_csv("parameters.csv") 
df_performance.to_csv("modelperformance.csv")

#Ablation studies
data2 = import_xlsx_data('upy AA input 3.xlsx')
df = data2.drop(data2.columns[0], axis=1)
X2, y2 = df.drop('Turbidity', axis=1), df[['Turbidity']]

df_results2, df_performance2 = XGBoost_regression(random_state, n_outer, n_inner, X2, y2, parameter_grid, iterations)
df_results2.to_csv("parameters ablation.csv") 
df_performance2.to_csv("modelperformance ablation.csv")
