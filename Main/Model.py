
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error, make_scorer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Normalizer
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectKBest, RFE, f_regression
import numpy as np
import pandas as pd
import time
from pathlib import Path
from Preprocessing import *
from BestModels import *


def train_test_split_data(df, size):
    X = df.drop('SalePrice', axis=1)
    Y = df.SalePrice
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=size, random_state=101)

    return X_train, X_test, y_train, y_test


def scale_data(x_train,x_test, numerical, scaler):
    if scaler == "RobustScaler":
        sc = RobustScaler()
    elif scaler == "MinMaxScaler":
        sc = MinMaxScaler()
    elif scaler == "Normalizer":
        sc = Normalizer()
    elif scaler == "StandardScaler":
        sc = StandardScaler()
    else:
        raise Exception("No such scaler: " + str(scaler) + ", please select form: StandardScaler, RobustScaler, MinMaxScaler, Normalizer.")
    x_train.loc[:, numerical] = sc.fit_transform(x_train.loc[:, numerical])
    x_test.loc[:, numerical] = sc.transform(x_test.loc[:, numerical])

    return x_train, x_test



def LinearRegression_model(X_train, X_test, y_train, y_test):
    # Linear Regression
    start = time.time()
    folds = KFold(n_splits=10, shuffle=True, random_state=101)
    lr = LinearRegression()
    scorer = make_scorer(rmsle, greater_is_better=False)
    scores = cross_val_score(lr, X_train, y_train, scoring=scorer, cv=folds)
    lr.fit(X_train, y_train)
    end = time.time()
    y_pred = lr.predict(X_test)
    result = rmsle(y_test, y_pred)
    best_parameters = ''
    best_score = abs(np.max(scores))
    fit_time = end - start
    return best_parameters, best_score, result, fit_time

def ElasticNet_model(X_train, X_test, y_train, y_test):
    #ElasticNet
    param = {
        'alpha': np.arange(0.1, 2, 0.1).tolist(),
        'l1_ratio': np.arange(0, 1, 0.05).tolist(),
        'selection': ['cyclic','random'],
        'warm_start': [True, False]
    }
    en= ElasticNet(random_state=101)
    scorer = make_scorer(rmsle, greater_is_better=False)

    start = time.time()
    en_search = RandomizedSearchCV(en, param, n_iter=25, cv=10, n_jobs=-1, random_state=101, scoring=scorer)
    en_search.fit(X_train, y_train)
    end = time.time()
    y_pred = en_search.predict(X_test)
    result = rmsle(y_test, y_pred)
    best_parameters = en_search.best_params_
    best_score = abs(en_search.best_score_)
    fit_time = end - start
    # print(fit_time)
    # print(best_parameters)
    # print(best_score)
    # print(result)
    return  best_parameters, best_score, result, fit_time


def RandomForestRegressor_model(X_train, X_test, y_train, y_test):
    #RandomForestRegressor

    param = {
        'n_estimators': [1, 4, 16, 32, 64, 100, 200, 400],
        'max_depth': [None, 16, 32, 64, 128],
        'min_samples_split': [2, 4, 8, 16],
        'max_features': ['auto', 'sqrt', 'log2'],
        'warm_start': [True,False]
    }

    rf= RandomForestRegressor(random_state=101)
    scorer = make_scorer(rmsle, greater_is_better=False)
    start = time.time()
    rf_search = RandomizedSearchCV(rf, param, n_iter=25, cv=10, n_jobs=-1, random_state=101, scoring=scorer)
    rf_search.fit(X_train, y_train)
    end = time.time()
    y_pred = rf_search.predict(X_test)
    result = rmsle(y_test, y_pred)
    best_parameters = rf_search.best_params_
    best_score = abs(rf_search.best_score_)
    fit_time = end - start
    # print(fit_time)
    # print(best_parameters)
    # print(best_score)
    # print(result)
    return best_parameters, best_score, result, fit_time


def GradientBoostingRegressor_model(X_train, X_test, y_train, y_test):
    #GradientBoostingRegressor

    param = {
        'loss': ['ls','lad','huber','quantile'],
        'learning_rate': [0.001,0.01,0.1,0.2],
        'n_estimators': [1, 4, 16, 32, 64, 100, 200, 400],
        'min_samples_split': [2, 4, 8, 16],
        'max_depth': [2, 3, 6, 12, 24, 48],
        'warm_start': [True, False]
    }

    gb= GradientBoostingRegressor(random_state=101)
    scorer = make_scorer(rmsle, greater_is_better=False)
    start = time.time()
    gb_search = RandomizedSearchCV(gb, param, n_iter=5, cv=10, n_jobs=-1, random_state=101, scoring=scorer)
    gb_search.fit(X_train, y_train)
    end = time.time()
    y_pred = gb_search.predict(X_test)
    result = rmsle(y_test, y_pred)
    best_parameters = gb_search.best_params_
    best_score = abs(gb_search.best_score_)
    fit_time = end - start
    # print(fit_time)
    # print(best_parameters)
    # print(best_score)
    # print(result)
    return best_parameters, best_score, result, fit_time

def AdaBoostRegressor_model(X_train, X_test, y_train, y_test):
    #AdaBoostRegressor

    param = {
        'n_estimators': [1, 4, 16, 32, 64, 100, 200, 400],
        'learning_rate': [0.001,0.01,0.1,1,2],
        'loss': ['linear','square','exponential'],
        'base_estimator': [DecisionTreeRegressor(max_depth=3), LinearRegression(), ElasticNet()],
    }

    ab= AdaBoostRegressor(random_state=101)
    scorer = make_scorer(rmsle, greater_is_better=False)
    start = time.time()
    ab_search = RandomizedSearchCV(ab, param, n_iter=5, cv=10, n_jobs=-1, random_state=101, scoring=scorer)
    ab_search.fit(X_train, y_train)
    end = time.time()
    y_pred = ab_search.predict(X_test)
    result = rmsle(y_test, y_pred)
    best_parameters = ab_search.best_params_
    best_score = abs(ab_search.best_score_)
    fit_time = end - start
    # print(fit_time)
    # print(best_parameters)
    # print(best_score)
    # print(result)
    return  best_parameters, best_score, result, fit_time


def KNeighnorsRegressor_model(X_train, X_test, y_train, y_test):
    # KNeighborsRegressor

    param = {
        'n_neighbors': [2, 3, 5, 8, 12, 16, 20],
        'p': np.arange(1, 2, 0.1).tolist(),
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto','ball_tree','kd_tree','brute'],
    }

    knn = KNeighborsRegressor()
    scorer = make_scorer(rmsle, greater_is_better=False)
    start = time.time()
    knn_search = RandomizedSearchCV(knn, param, n_iter=25, cv=10, n_jobs=-1, random_state=101, scoring=scorer)
    knn_search.fit(X_train, y_train)
    end = time.time()
    y_pred = knn_search.predict(X_test)
    result = rmsle(y_test, y_pred)
    best_parameters = knn_search.best_params_
    best_score = abs(knn_search.best_score_)
    fit_time = end - start
    # print(fit_time)
    # print(best_parameters)
    # print(best_score)
    # print(result)
    return best_parameters, best_score, result, fit_time


def SVR_model(X_train, X_test, y_train, y_test):
    # SVR

    param = {
        'kernel': ['linear','poly','rbf','sigmoid'],
        'C': [0.01,0.1,1,2,4,8,12,16,32,64],
        'shrinking': [True, False],
    }

    svr = SVR()
    scorer = make_scorer(rmsle, greater_is_better=False)
    start = time.time()
    svr_search = RandomizedSearchCV(svr, param, n_iter=10, cv=10, n_jobs=-1, random_state=101, scoring=scorer)
    svr_search.fit(X_train, y_train)
    end = time.time()
    y_pred = svr_search.predict(X_test)
    result = rmsle(y_test, y_pred)
    best_parameters = svr_search.best_params_
    best_score = abs(svr_search.best_score_)
    fit_time = end - start
    return best_parameters, best_score, result, fit_time


def MLPRegressor_model(X_train, X_test, y_train, y_test):
    # MLPRegressor

    param = {
        'activation': ['identity','logistic','tanh','relu'],
        'solver': ['lbfgs','adam'],
        'alpha': [0.00001,0.0001,0.001,0.01,0.1],
        'learning_rate': ['constant','invscaling','adaptive'],
        'warm_start': [True, False],
        'hidden_layer_sizes': [(1,), (2,), (4,), (8,), (16,), (32,), (64,), (100,), (128,), (264,)]
    }

    mlp = MLPRegressor(random_state=101)
    scorer = make_scorer(rmsle, greater_is_better=False)
    start = time.time()
    mlp_search = RandomizedSearchCV(mlp, param, n_iter=10, cv=10, n_jobs=-1, random_state=101, verbose=1,scoring=scorer)
    mlp_search.fit(X_train, y_train)
    end = time.time()
    y_pred = mlp_search.predict(X_test)
    result = rmsle(y_test, y_pred)
    best_parameters = mlp_search.best_params_
    best_score = abs(mlp_search.best_score_)
    fit_time = end - start
    # print(fit_time)
    # print(best_parameters)
    # print(best_score)
    # print(result)
    return best_parameters, best_score, result, fit_time



def rmsle(y_test, y_pred):
    y_pred = np.abs(y_pred)
    error = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return error



def get_result_for_models(df, scaler, numerical_features, test_size, outlier_handling):
    #Done: KNeighnorsRegressor_model, ElasticNet_model, RandomForestRegressor_model, SVR_model, GradientBoostingRegressor_model, AdaBoostRegressor_model, LinearRegression_model
    #TODO:  MLPRegressor_model
    all_models = [ElasticNet_model, RandomForestRegressor_model,GradientBoostingRegressor_model,
                  AdaBoostRegressor_model,KNeighnorsRegressor_model,SVR_model,MLPRegressor_model, LinearRegression_model]
    all_models = [SVR_model]

    X_train, X_test, y_train, y_test = train_test_split_data(df, size=test_size)

    if scaler != "None":
        X_train, X_test = scale_data(X_train, X_test, numerical=numerical_features, scaler=scaler)


    data = []
    for model in all_models:
        best_parameters, rmsle_validation, rmsle_test, fit_time = model(X_train, X_test, y_train, y_test)
        new_model = {'Model':str(model).split(' ')[1].split('_')[0], 'HyperParameters': best_parameters, 'Scaler': scaler,
                     'Test Size' :test_size, 'Outlier Handling' : outlier_handling, 'Fit Time(s)': fit_time,
                     'Validation RMSLE':rmsle_validation, 'Test RMSLE':rmsle_test}
        data.append(new_model)
        print(str(model).split(' ')[1].split('_')[0] + '_' + str(scaler) + '_' + str(test_size) + '_' + str(
            outlier_handling))

    report_df = pd.DataFrame(data=data)


    name = '../Report/Model/Model Performances.csv'

    if sum(1 for _ in Path("../Report/Model/").glob('**/*.csv')) != 0:
        all_other_datasets = pd.read_csv('../Report/Model/Model Performances.csv')
        all_models = pd.concat([report_df, all_other_datasets])
    else:
        all_models = report_df

    all_models = all_models.sort_values('Test RMSLE',ascending=True)
    all_models.to_csv(name, index=False)

def create_submission_for_top3_models(test, train, nftrain, scaler, index):
#{'n_estimators': 32, 'loss': 'exponential', 'learning_rate': 0.01, 'base_estimator': ElasticNet()}
#{'warm_start': True, 'selection': 'cyclic', 'l1_ratio': 0.9500000000000001, 'alpha': 0.30000000000000004}
    model = LinearRegression()

    X_train = train.drop('SalePrice', axis=1)
    y_train = train[['SalePrice']]
    X_test = test



    X_train, X_test = drop_columns_not_in_test(X_train, X_test)

    if scaler != "None":
        X_train, X_test = scale_data(X_train, X_test, numerical=nftrain, scaler=scaler)

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    y_pred = model.predict(X_test)
    fit_time = end - start
    print(fit_time)

    data = {
        'Id': index,
        'SalePrice': list(y_pred)
    }
    submit_df = pd.DataFrame(data=data)


    submit_df.to_csv('../Submissions/LinearRegression.csv', index=False)



def drop_columns_not_in_test(X_train,X_test):
    train_columns = list(X_train.columns)
    test_columns = list(X_test.columns)

    wanted_cols = list(set(train_columns) & set(test_columns))

    X_train = X_train[wanted_cols]
    X_test = X_test[wanted_cols]

    return  X_train, X_test


def feature_selection_selectKbest():
    best_model1, best_scaler, best_outlier_selection, best_test_size = best_models()

    import warnings
    warnings.filterwarnings("ignore")

    data = []
    to_try = [1,5,10,15,20,25,30,35,40,50,60,70,80,100]
    for i in range(0, len(to_try)):
        best_model, best_scaler, best_outlier_selection, best_test_size = best_models()
        for j in range(0, len(best_model)):
            df = pd.read_csv('../data/train.csv')
            df = handle_missing_values(df)
            df, numerical_features = handle_outliers(df, method=best_outlier_selection[j])
            df = transform_data(df)

            X_train, X_test, y_train, y_test = train_test_split_data(df, size=best_test_size[j])

            if best_scaler[j] != "None":
                X_train, X_test = scale_data(X_train, X_test, numerical=numerical_features, scaler=best_scaler[j])

            model_cv = best_model[j]
            fs = SelectKBest(f_regression, k=to_try[i])
            fs.fit(X_train, y_train)
            X_train_fs = fs.transform(X_train)
            X_test_fs = fs.transform(X_test)

            start = time.time()
            model_cv.fit(X_train_fs, y_train)
            end = time.time()
            y_pred = model_cv.predict(X_test_fs)
            result = rmsle(y_test, y_pred)
            mdl = str(model_cv).split('(')
            fit_time = end - start
            new_val = {'Model': mdl[0], 'HyperParemeters': '('+mdl[1], 'Scaler': best_scaler[j], 'Test Size': best_test_size[j], 'Outlier Handling':best_outlier_selection[j],
                       'Fit Time(s)': fit_time, 'Top Attributes': str(to_try[i]), 'Test RMSLE': result}
            data.append(new_val)
            print('Model:' + str(mdl[0]) + ', Top:'+ str(to_try[i]))


    report_df = pd.DataFrame(data=data)
    name = '../Report/FeatureSelection/FeatureSelection_SelectKBest.csv'

    if sum(1 for _ in Path("../Report/FeatureSelection/").glob('FeatureSelection_SelectKBest.csv')) != 0:
        all_other_datasets = pd.read_csv('../Report/FeatureSelection/FeatureSelection_SelectKBest.csv')
        all_models = pd.concat([report_df, all_other_datasets])
    else:
        all_models = report_df

    all_models = all_models.sort_values('Test RMSLE', ascending=True)
    all_models.to_csv(name, index=False)

def feature_selection_RFE():

    data = []
    to_try = [1, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70]
    for i in range(0, len(to_try)):

        best_model, best_scaler, best_outlier_selection, best_test_size = best_models()

        for j in range(0, len(best_model)):
            model_cv = best_model[j]
            if j == 3 or j == 5:
                continue
            df = pd.read_csv('../data/train.csv')
            df = handle_missing_values(df)
            df, numerical_features = handle_outliers(df, method=best_outlier_selection[j])
            df = transform_data(df)

            X_train, X_test, y_train, y_test = train_test_split_data(df, size=best_test_size[j])

            if best_scaler[j] != "None":
                X_train, X_test = scale_data(X_train, X_test, numerical=numerical_features, scaler=best_scaler[j])


            start = time.time()
            rfe = RFE(estimator=model_cv, n_features_to_select=to_try[i], step=15)
            # fit the model
            rfe.fit(X_train, y_train)
            end = time.time()
            y_pred = rfe.predict(X_test)
            result = rmsle(y_test, y_pred)
            mdl = str(model_cv).split('(')
            fit_time = end - start
            new_val = {'Model': mdl[0], 'HyperParemeters': '(' + mdl[1], 'Scaler': best_scaler[j],
                       'Test Size': best_test_size[j], 'Outlier Handling': best_outlier_selection[j],
                       'Fit Time(s)': fit_time, 'Top Attributes': str(to_try[i]), 'Test RMSLE': result}
            data.append(new_val)
            print('Model:' + str(mdl[0]) + ', Top:' + str(to_try[i]) + ', Time:' + str(fit_time))


    report_df = pd.DataFrame(data=data)
    name = '../Report/FeatureSelection/FeatureSelection_RFE.csv'

    if sum(1 for _ in Path("../Report/FeatureSelection/").glob('FeatureSelection_RFE.csv')) != 0:
        all_other_datasets = pd.read_csv('../Report/FeatureSelection/FeatureSelection_RFE.csv')
        all_models = pd.concat([report_df, all_other_datasets])
    else:
        all_models = report_df

    all_models = all_models.sort_values('Test RMSLE', ascending=True)
    all_models.to_csv(name, index=False)

def create_submission_for_bestfeatures(test, train, nftrain, index):
    best_model, best_scaler, best_outlier_selection, best_test_size = best_models()
    model = best_model[1]
    scaler = best_scaler[1]

    X_train = train.drop('SalePrice', axis=1)
    y_train = train[['SalePrice']]
    X_test = test



    X_train, X_test = drop_columns_not_in_test(X_train, X_test)

    if scaler != "None":
        X_train, X_test = scale_data(X_train, X_test, numerical=nftrain, scaler=scaler)

    start = time.time()
    rfe = RFE(estimator=model, n_features_to_select=70, step=15)
    rfe.fit(X_train, y_train)
    end = time.time()
    y_pred = rfe.predict(X_test)
    fit_time = end - start
    print(fit_time)

    data = {
        'Id': index,
        'SalePrice': list(y_pred)
    }
    submit_df = pd.DataFrame(data=data)


    submit_df.to_csv('../Submissions/ElasticNet_top70features.csv', index=False)