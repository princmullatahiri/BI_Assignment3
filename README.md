# BI_Assignment3

## Dataset Selection

We got the dataset from a Kaggle competition: [Ames Housing Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview/description).
It consist of a Regression problem where we should predict House Prices. It has 79 attributes and 1460 instances.

## Preprocessing

### Handling missing values

For handling missing values we replaced the missing values with most frequent result if variable was categorical and with median if it was numeric.

### Handling outliers

* Log: Log transform values to reduce the impact of outliers
* Yeo-Jonson: Use Yeo-Jonson transformation to reduce impact of outliers
* Replace: Replace outliers with random 2.5% of upper or lower values which are not outliers

### Transforming dataset

* Use dummy encoding for all
* Turn some ordinal values to numeric

## Modeling

We used 10 fold cross validation in order to find the best hyperparameters.

Models:
* RandomForestRegressor
* ElasticNet
* GradientBoostingRegressor
* MLPRegressor
* SVR
* LinearRegression
* KNeighborsRegressor
* AdaBoostRegressor
* XGBRegressor
* Keras Sequential Neural Network

Metrics that were used were: RMSLE, RMSE, MAE, R2.
The best models were used to perform feature selection, PCA, other transformations and to submit results in competition.
All the results can be found in: *Report\FeatureSelection* and *Report\Model* 
The best model was XGBRegressor with these hyperparameters *subsample= 0.7, objective= 'reg:squarederror', n_estimators= 64, min_child_weight= 3, max_depth= 6, learning_rate= 0.1, colsample_bytree= 0.6*  using MinMaxScaler and Log transformation and using for testing 10% of values with all categorical variables turned into dummy variables with a RMSLE of 0.14059.


## Code

### Main.py
 
Main contains all the transformations, outlier handling, model selection that we could call. 
We first need to install requirements using the command below:
```
pip install -r /../requirements.txt
```

### DataAnalysis.py

This python file contains all the visualizations that we did in order to analyse data. All the visualiztions can be found under *visualizations\Data Analysis*.

### Preprocessing.py

This python file contains methods for handling missing values, handling outliers, transforming data.

### Model.py

This python file contains methods for identifying best hyperparameters for each model, for finding the best method for handling outliers and for saving results of our models.

### BestModel.py

Contains all the best model with best hyperparameters and best scalers, methods for handling outliers, test size.

### VisualizationOfResults.py

Contains all visualizations of the model's scores. All the visualiztions can be found under *visualizations\Results*.
