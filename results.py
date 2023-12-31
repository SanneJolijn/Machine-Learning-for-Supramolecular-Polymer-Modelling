# Import necessary packages
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import math 

# Define coefficients and feature importance data for various experiments
coeficients_linear_regression = [[0.017, 0.034, 0.029, 0.032, 0.793, 0.013, 0.018, 0.228, 0.022], 
            [0.023, 0.033, 0.009, 0.031, 0.785, 0.009, 0.014, 0.216, 0.022],
            [0.020, 0.029, 0.019, 0.032, 0.786, 0.011, 0.017, 0.221, 0.023],
            [0.019, 0.034, 0.020, 0.034, 0.791, 0.011, 0.017, 0.223, 0.022],
            [0.010, 0.044, 0.028, 0.040, 0.761, 0.020, 0.026, 0.231, 0.030],
            [0.019, 0.034, 0.018, 0.032, 0.790, 0.011, 0.018, 0.227, 0.022],
            [0.017, 0.035, 0.022, 0.034, 0.786, 0.013, 0.019, 0.221, 0.024],
            [0.020, 0.034, 0.020, 0.031, 0.787, 0.010, 0.017, 0.223, 0.022],
            [0.017, 0.035, 0.023, 0.033, 0.788, 0.013, 0.019, 0.228, 0.024],
            [0.021, 0.033, 0.019, 0.032, 0.783, 0.011, 0.016, 0.224, 0.022]]

coeficients_svr = [[0.09978116, -0.03753138,  0.01448566, -0.023004, 0.71753661, 0, -0.00940228, 0.222691, -0.05170479],
            [0.10014908, -0.00677087, -0.09327118, 0, 0.71626994, 0, -0.00460514, 0.19180554, -0.00373006],
            [0.09966209, 0.03000797, 0, 0.00400798, 0.62926087, 0.00502062, 0.01634841, 0.22021289, -0.02797938],
            [0.09919484, 0.03103331, 0, 0.00519194, 0.62879258, 0.00635665, 0.01703331, 0.22046467, -0.02663183],
            [0.10002341, 0.03910181, 0.00261215, 0.01304008, 0.62017624, 0.01374222, 0.02163185, 0.21659195, -0.00305494],
            [0.09939606, 0.0287137, 0, 0.00236237, 0.63112816, 0.00360256, 0.01436235, 0.22057258, -0.02950618],
            [0.10012831, 0.03122354, 0, 0.00525634, 0.6275932, 0.00622354, 0.01764516, 0.21986911, -0.02703274],
            [0.09943595, 0.03189392, 0, 0.00507228, 0.62808679, 0.00211976,  0.01776551, 0.22048847, -0.02623449],
            [0.09963002, 0.0136952, 0, 0.0065105, 0.62701771, 0.00757914, 0.01816406, 0.2204224,  -0.03374891],
            [0.09992006, 0.0312042077,  0, -0.000469381850,  0.627830657,   0.00609859565,  0.0173441183,  0.220344246, -0.0271674634]]

feature_importance = [[0.00721734, 0.00692904, 0.00758401, 0.83145446, 0.00768912, 0.00749472, 0.12453604, 0.00709533],
                   [0.02298941, 0.04722924, 0.02433802, 0.7420176, 0.04351439, 0.03633617, 0.06260879, 0.02096642],
                   [0.00801976, 0.01087282, 0.00596275, 0.92072743, 0.01159769, 0.00723789, 0.02905534, 0.00652629],
                   [0.00565463, 0.00861272, 0.00469919, 0.9059558,  0.00721472, 0.00316089, 0.05869493, 0.00600712],
                   [0.00822646, 0.01407445, 0.00427639, 0.86565477, 0.0053404,  0.00633033, 0.09084102, 0.00525622],
                   [0.00510205, 0.01421082, 0.00796402, 0.90409124, 0.01198071, 0.00919576, 0.04013713, 0.00731823],
                   [0.01093611, 0.04305977, 0.03645724, 0.77581805, 0.00787413, 0.01217457, 0.07151715, 0.04216306],
                   [0.01377946, 0.04057097, 0.03414657, 0.76064855, 0.03776582, 0.02262599, 0.06265761, 0.02780512],
                   [0.02544933, 0.04073162, 0.02112175, 0.777679,   0.04362853, 0.02630638, 0.0481368,  0.01694654],
                   [0.00434185, 0.00780581, 0.00539173, 0.9095364,  0.00444814, 0.00386123, 0.05895283, 0.00566204]]

feature_importance_ablation_studies = [[0.08004041, 0.07108837, 0.06128893, 0.04922848, 0.04259803, 0.63381195, 0.0619438],
                [0.05649197, 0.11088742, 0.03724211, 0.02489162, 0.02892691, 0.7122165, 0.02934355],
                [0.06431913, 0.0466296 , 0.0322955,  0.04405026, 0.01672694, 0.7554531, 0.04052549],
                [0.04277706, 0.17192034, 0.05678926, 0.03783668, 0.03089868, 0.6202233, 0.03955472],
                [0.09945171, 0.12888922, 0.07286862, 0.06323439, 0.05883329, 0.51640123, 0.06032152],
                [0.08606993, 0.04189545, 0.04688812, 0.03892288, 0.03052532, 0.7065161, 0.04918225],
                [0.07463723, 0.14209883, 0.04473987, 0.0159513,  0.0382245,  0.63684, 0.0475083],
                [0.02876752, 0.08644741, 0.02208426, 0.01449958, 0.01151362, 0.819626, 0.01706169],
                [0.05694214, 0.09116143, 0.05277887, 0.02747541,0.03749828, 0.7035312, 0.03061272 ],
                [0.0833538,  0.07766573, 0.09305561, 0.07297512, 0.0566162,  0.5543745, 0.06195907]]

coeficients_linear_regression_ablation_studies = [[0.0076103, 0.04147688, 0.03837034, 0.04252937, 0.0170589,  0.02378307, 0.23652254,  0.03167009],
                     [0.00816552, 0.04054806, 0.03715931, 0.04192614, 0.01681316, 0.02292773, 0.23854254,  0.03276265],
                     [0.00949428, 0.041138,   0.0278983,  0.04077883, 0.01654037, 0.02296836, 0.23030252,  0.03299427],
                     [0.00874713, 0.03629811, 0.03785212, 0.04299056, 0.01695112, 0.02242,    0.23268054,  0.03320679],
                     [0.00697318, 0.03953071, 0.03905844, 0.04523847, 0.01673403, 0.02330673, 0.2421621,  0.03237397],
                     [0.00837832, 0.04127967, 0.03918172, 0.04259702, 0.01631835, 0.02126068, 0.23686783,  0.03281264],
                     [0.00520443, 0.0428812,  0.04046464, 0.04488508, 0.02093308, 0.02510874, 0.23251787,  0.03496726],
                     [0.00634552, 0.04203309, 0.03880572, 0.0429509,  0.01710068, 0.02519119, 0.24026246,  0.03460979],
                     [0.00669264, 0.04061742, 0.03857498, 0.04317722, 0.01842346, 0.02355033, 0.23839251,  0.03310444],
                     [0.00702926, 0.04116894, 0.03788628, 0.04314097, 0.01714556, 0.02343239, 0.23827585,  0.03451518]]

coeficients_svr_ablation_studies = [[0.10000427, 0, 0.0320141, 0, 0, -0.00032014, 0.2049795, -0.00172965],
                     [0.10000427, 0, 0.0320141, 0, 0, -0.00032014, 0.2049795, -0.00172965],
                     [0.1000192,  0, 0,         0, 0, 0,           0.1919808, -0.00191981],
                     [0.10000427, 0, 0.0320141, 0, 0, -0.00032014, 0.2049795, -0.00172965],
                     [0.1000205,  0, 0.032,     0, 0, 0,           0.2049795, -0.0020498],
                     [0.1000096, 0, 0.034, 0,          0,         -0.001,      0.1919908,  -0.00091991],
                     [0.10000427, 0, 0.0320141, 0, 0, -0.00032014, 0.2049795, -0.00172965],
                     [0.1000096, 0,          0.032,       0,          0,         -0.00091991,  0.1919908,  -0.001],
                     [0.10000427, 0, 0.0320141, 0, 0, -0.00032014, 0.2049795, -0.00172965],
                     [0.1000096, 0,          0.032,       0,          0,         -0.00091991,  0.1919908,  -0.001]]

#Plot results of feature importance and coefficients of regression
features = [coeficients_linear_regression, 
            coeficients_svr_ablation_studies, 
            coeficients_linear_regression_ablation_studies, 
            feature_importance_ablation_studies, 
            feature_importance, 
            coeficients_svr]

labels = [[0,1,2,3,4,5,6,7,8],
          [0,1,2,3,5,6,7,8],
          [0,1,2,3,5,6,7,8],
          [1,2,3,5,6,7,8],
          [1,2,3,4,5,6,7,8],
          [0,1,2,3,4,5,6,7,8]]

titles = ['Hight coefficients of linear regression',
          'Hight coefficients of svr (ablation studies)',
          'Hight coefficients of linear regression (ablation studies)',
          'Feature importances in XGBoost model (ablation studies)',
          'Feature importances in XGBoost model',
          'Hight coefficients of svr']

for i in range(len(features)):
    sns.heatmap(features[i], xticklabels=labels[i])
    plt.title(titles[i])
    plt.xlabel('component number')
    plt.ylabel('number outer loop')
    plt.show()

# Clean and process results data frames
data = pd.read_csv("parameters.csv")
df = data.dropna()

df_SVR = pd.read_csv("regression SVR results.csv")
df_SVR2 = df_SVR.dropna()

# Get the best model and its parameter values for XGBoost
df_rank = df[['mean_test_score']]
index_maxvalue = df_rank.idxmax(axis=0)
print(index_maxvalue)
result = df.iloc[[8272]]
result.to_csv("best model.csv") 

# Get the best model and its parameter values for SVR
df_rank2 = df_SVR2[['mean_test_score']]
index_maxvalue2 = df_rank2.idxmax(axis=0)
print(index_maxvalue2)
result2 = df_SVR2.iloc[[15]]
result2.to_csv("best model SVR.csv") 

# Define parameter lists for boxplots and heatmaps
params = ['param_subsample', 'param_colsample_by_tree', 'param_n_estimators', 'param_learning_rate', 'param_max_depth', 'param_reg_alpha']
params2 = ['param_gamma', 'param_C']

# Create boxplots for hyperparameters
def boxplots_results(params, df):
    # Loop through hyperparameters and create boxplots
    for parameter in params:
        unique_vals = df[parameter].unique()
        unique_vals.sort()
        data = {}
        for val in unique_vals:
            df1 = df['mean_test_score'][df[parameter] == val].sort_values(ascending=False)
            df2 = df1.iloc[:25]
            data[str(val)] = df2.to_list()
        fig, ax = plt.subplots()
        ax.boxplot(data.values())
        ax.set_xticklabels(data.keys())
        ax.set_title(parameter)
        ax.set(xlabel='hyperparameter value', ylabel='mean test score')
        plt.show()

    return None

boxplots_results(params, df)
boxplots_results(params2, df_SVR2)

# Create heatmaps for hyperparameters
def heatmaps_results(params, df):
    # Loop through hyperparameters and create heatmaps
    for parameter in params:
        unique_vals = df[parameter].unique()
        unique_vals.sort()
        means = []
        labels = []
        dict_param = {}
        for val in unique_vals:
            df1 = df['mean_test_score'][df[parameter] == val].sort_values(ascending=False)
            df2 = df1.iloc[:25]
            mean = df2.mean()
            dict_param[val] = mean
            means.append(mean)
            labels.append(val)
        array = [means]
        ax = sns.heatmap(array, xticklabels=labels)
        ax.set(xlabel='hyperparameter value', ylabel=None)
        ax.set_title(parameter)
        ax.tick_params(left=False)
        ax.set_yticks([])
        plt.show()

    return None

heatmaps_results(params, df)
heatmaps_results(params2, df_SVR2)

# %%
