from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

def best_models(i=None):
    best_model = [
        RandomForestRegressor(random_state=101,warm_start=False, n_estimators=200, min_samples_split=8, max_features='auto', max_depth=32),
        ElasticNet(random_state=101,warm_start=True, selection='cyclic', l1_ratio=0.95, alpha=0.3),
        GradientBoostingRegressor(random_state=101,warm_start=False, n_estimators=400, min_samples_split=16, max_depth=24, loss='huber', learning_rate=0.2),
        AdaBoostRegressor(random_state=101, n_estimators=32, loss='exponential', learning_rate=0.01, base_estimator=ElasticNet()),
        LinearRegression(),
        KNeighborsRegressor(weights='distance', p=1.1, n_neighbors=5, algorithm='kd_tree'),
        SVR(shrinking=False, kernel='linear', C=4),
        MLPRegressor(random_state=101,warm_start=False, solver='adam', learning_rate='adaptive',hidden_layer_sizes=(264,),alpha=0.1,activation='identity'),
        XGBRegressor(random_state=101,subsample= 0.7, objective= 'reg:squarederror', n_estimators= 64, min_child_weight= 3, max_depth= 6, learning_rate= 0.1, colsample_bytree= 0.6)
    ]

    best_scaler = [
        'RobustScaler','MinMaxScaler','MinMaxScaler','StandardScaler','None','StandardScaler','None','None','MinMaxScaler'
    ]


    best_outlier_selection = [
        'YeoJonson','Replace','YeoJonson','Replace','YeoJonson','Replace','Log','Replace','Log'
    ]

    best_test_size = [
        0.1,0.1,0.1,0.3,0.1,0.1,0.1,0.3,0.1
    ]
    if i == None:
        return best_model, best_scaler, best_outlier_selection, best_test_size
    else:
        return best_model[i], best_scaler[i], best_outlier_selection[i], best_test_size[i]



