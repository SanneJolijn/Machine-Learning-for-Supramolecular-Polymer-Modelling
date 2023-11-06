# Machine Learning for Supramolecular Polymer Modelling
## Introduction
This repository contains code for building a machine learning model to predict properties, turbidity in this case study, based on the composition of supramolecular polymers (SMPs). Supramolecular Polymers are a promising class of materials for biomedical applications. They are characterized by their dynamic reversibility, stimuli responsiveness, and ease of functionalization and their properties make them ideal for a variety of biomedical applications including drug delivery, tissue engineering, and self-healing materials. However, the complex structure and interactions of Supramolecular Polymers can make it difficult to accurately model them.  Machine learning could be a valuable tool to model Supramolecular polymers and predict their properties, such as their turbidity. This thesis proposes a research study to investigate the use of Machine Learning to predict the properties of Supramolecular Polymers. The first case study focussed on predicting the turbidity of Supramolecular Polymers based on their composition. Turbidity is an important property for many applications, such tissue engineering. We used a variety of Machine Learning methods to predict the turbidity of Supramolecular Polymers. The evaluation of the model performances hinges on two critical metrics: the R-squared coefficient and the Root Mean Squared Error. The results indicate that machine learning models are able to accurately model Supramolecular Polymers for turbidity prediction. With R-squared values above 80\% and Root Mean Squared Errors below 0.1. The results will be used in future research using Machine Learning for further Supramolecular Polymer modelling. The research will also provide insights into the relationship between the composition and properties of Supramolecular Polymers.

## Table of contents
- Data overview
- Installation
- Usage
- Contributing
- License

## Data Overview
The dataset used in this project consists of 192 data points, where turbidity measurements of SMPs are taken along with information about the monomer units used and their concentrations. It also includes information about the concentration of the 8 selected compounds which were used to create the supramolecular polymer. Notably, the concentrations vary between 1 mM, 0.1 mM, and 0.01 mM, with volumes remaining constant. This dataset was also modified to include values of zero for compounds not present in each supramolecular polymer. This results in a dataset with 10 columns, including sample names, turbidity measurements, and concentration values for each compound.

## Installation
Follow the installation commands below.

$ git clone https://github.com/SanneJolijn/Machine-Learning-for-Supramolecular-Polymer-Modelling.git <br />
$ cd Machine-Learning-for-Supramolecular-Polymer-Modelling <br />
$ npm install

## Usage
To use this project, follow these steps:

**Install Packages:**
Make sure you have the necessary packages installed. If you haven't already, you can install them with:

**Data Preparation:**
Before running the code, you need to ensure that you have the dataset in the correct format. Make sure you have the dataset file (e.g., dataset.csv) in the project directory.

**Run the Model:**
You can run the model using the provided scripts. For example, to train the XGBoost regression model, use:

python train_xgboost_model.py
For other regression models (linear regression or SVR), you can run the corresponding scripts.


