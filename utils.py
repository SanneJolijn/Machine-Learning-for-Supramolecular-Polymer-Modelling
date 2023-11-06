# Import necessary packages
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Define functions to import data from CSV and XLSX files
def import_csv_data(filename):
    # Read data from a CSV file and remove columns with NaN values
    df = pd.read_csv(filename)
    df2 = df.dropna(axis=1)

    return df2

def import_xlsx_data(filename):
    # Read data from an XLSX file and remove columns with NaN values
    df = pd.read_excel(filename)
    df2 = df.dropna(axis=1)
    
    return df2

# Define a function to divide data into categorical and numerical columns
def divide_data(dataframe):
    cat_cols = dataframe.select_dtypes(include=['object']).columns
    num_cols = dataframe.select_dtypes(include=np.number).columns.tolist()

    return cat_cols, num_cols

# Define a function to create a bar plot and calculate skewness
def barplot_and_skew(columns, dataframe):
    col = columns[0]
    print(col)
    skew = round(dataframe[col].skew(), 2)
    fig = plt.figure(figsize = (15, 4))
    plt.subplot(1, 2, 1)
    dataframe[col].hist(grid=False)
    plt.ylabel('count')
    fig.xlabel('turbidity')
    plt.subplot(1, 2, 2)
    sns.boxplot(x=dataframe[col])
    
    return skew, fig

# Define a function to create a heatmap of correlation
def heatmap(dataframe):
    fig = plt.figure(figsize=(12, 7))
    sns.heatmap(dataframe.corr(), annot = True, vmin = -1, vmax = 1)
    
    return fig

# Define a function to split and scale data
def split_and_scale(dataframe, category_name):
    X, y = dataframe.drop(category_name, axis=1), dataframe[[category_name]]
    X = StandardScaler().fit_transform(X)

    return X, y

# Define a function to perform PCA (Principal Component Analysis) testing
def pca_test(X):
    pca_test = PCA()
    pca_test.fit(X)
    eigenvalues = pca_test.explained_variance_
    explained_variance = pca_test.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    components = np.arange(1, len(explained_variance) + 1)

    fig = plt.figure(figsize = (15, 4))
    plt.bar(components, cumulative_variance)
    plt.axhline(y = 0.8, color = 'r', linestyle = ':')

    # Set plot labels and title
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance for Principal Components')
    
    return fig

# Define a function to perform PCA
def pca(X, n_components):
    pca = PCA(n_components=6)
    pca_turbidity = pca.fit_transform(X)
    
    return pca, pca_turbidity

# Define a function to create a loading plot for PCA
def loading_plot(dataframe, pca):
    variable_names = dataframe.columns
    loadings = pca.components_
    n_features = pca.n_features_
    feature_names = dataframe.columns
    pc_list = [f'PC{i}' for i in list(range(1, n_features + 1))]
    pc_loadings = dict(zip(pc_list, loadings))
    
    loadings_df = pd.DataFrame.from_dict(pc_loadings)
    loadings_df['feature_names'] = feature_names
    loadings_df = loadings_df.set_index('feature_names')
    loadings_df

    xs = loadings[0]
    ys = loadings[1]

    fig = plt.figure(figsize = (15, 4))

    for i, varnames in enumerate(variable_names):
        plt.scatter(xs[i], ys[i], s=200)
        plt.text(xs[i], ys[i], varnames)
    
    xticks = np.linspace(-0.6, 0.6, num=5)
    yticks = np.linspace(-0.6, 0.6, num=5)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
    plt.title('2D Loading plot')
    plt.grid(True)
    
    return fig

# Define a function to create a score plot for PCA
def score_plot(pca):
    fig = px.scatter(pca, 
                 x=0, y=1, 
                 title='PCA score plot', 
                 labels={'0':'PC1 scores', '1':'PC2 scores'})

    return fig

# Define a function to create a grouped score plot for PCA
def grouped_score_plot(pca, dataframe, color_map, sort_by, legend):
    scores = pca[:, :2]
    fig = plt.figure(figsize=(20, 10))
    types = sorted(dataframe[sort_by].unique())

    for t in types:
        subset_scores = scores[dataframe[sort_by] == t]
        plt.scatter(subset_scores[:, 0], 
                    subset_scores[:, 1], 
                    color=color_map[str(t)], 
                    label=t)

    plt.xlabel('PC1 Scores')
    plt.ylabel('PC2 Scores')
    plt.title('PCA Score Plot')
    plt.legend(legend)
    plt.grid(True)
    
    return fig

# Define a function to visualize regression results
def vis_regression(X_train, X_test, y_train, y_pred):
    fig = plt.figure(figsize=(12,7))
    plt.scatter(X_train, y_train, color='g')
    plt.plot(X_test, y_pred, color='k')

    return fig
