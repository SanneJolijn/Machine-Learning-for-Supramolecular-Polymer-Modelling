#%%
#Import all necessary packages and files
import matplotlib.pyplot as plt
import pandas as pd

from utils import (import_xlsx_data, divide_data,
                   barplot_and_skew, heatmap,
                   split_and_scale, pca_test, 
                   pca, loading_plot, score_plot, 
                   grouped_score_plot)

#%%
#Load data set and print all usefull information
data = import_xlsx_data('upy AA input.xlsx')
data.info()
data.nunique()
data.isnull().sum()

#%%
#Drop the column with the experiment names
data2 = data.drop(data.columns[0], axis=1)
data2.describe()

# %%
#Visualize dataset
cat_cols, num_cols = divide_data(data2)

f = barplot_and_skew(num_cols, data2)
plt.show()

#%%
#Check for correlation
g = heatmap(data2)
plt.show()

#%%
#Perform PCA
X, y = split_and_scale(data2, 'Turbidity')

h = pca_test(X)
plt.show()

pca, pca_turbidity = pca(X, 6)

# %%
#Show the loading plot
X_new = pd.DataFrame(X)

i = loading_plot(X_new, pca)
plt.show()

# %%
#Show the score plot
j = score_plot(pca_turbidity)
j.show()

# %%
#Import the dataset with extra information
data3 = import_xlsx_data('upy AA input 2.xlsx')

# %%
#Show score plot grouped by major compound
color_map1 = {'c1': 'red', 'c2': 'blue', 'c3': 'green',
             'c4': 'orange', 'c5': 'purple', 'c6': 'black',
             'c7': 'pink', 'c8': 'lime'}
legend1 = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8']
k = grouped_score_plot(pca_turbidity, data3, color_map1, 'Main component', legend1)
plt.show()

# %%
#Show score plot grouped by turbidity 'highness'
color_map2 = {'w': 'red', 'x': 'blue', 'y': 'green', 'z': 'orange'}
legend2 = ['w', 'x', 'y', 'z']
l = grouped_score_plot(pca_turbidity, data3, color_map2, 'HL_turbidity', legend2)
plt.show()

#%%
#Show score plot grouped by letter?
color_map3 = {'A': 'red', 'B': 'blue', 'C': 'green',
             'D': 'orange', 'E': 'purple', 'F': 'black',
             'G': 'pink', 'H': 'lime', 'I': 'grey'}
legend3 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
m = grouped_score_plot(pca_turbidity, data3, color_map3, 'Letter', legend3)
plt.show()
