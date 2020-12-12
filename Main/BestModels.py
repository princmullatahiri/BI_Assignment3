from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


def best_models():
    best_model = [
        RandomForestRegressor(warm_start=False, n_estimators=200, min_samples_split=8, max_features='auto', max_depth=32),
        ElasticNet(warm_start=True, selection='cyclic', l1_ratio=0.95, alpha=0.3),
        GradientBoostingRegressor(warm_start=False, n_estimators=400, min_samples_split=16, max_depth=24, loss='huber', learning_rate=0.2),
        AdaBoostRegressor(n_estimators=32, loss='exponential', learning_rate=0.01, base_estimator=ElasticNet()),
        LinearRegression(),
        KNeighborsRegressor(weights='distance', p=1.1, n_neighbors=5, algorithm='kd_tree'),
        SVR(shrinking=False, kernel='linear', C=4),
        MLPRegressor(warm_start=False, solver='adam', learning_rate='adaptive',hidden_layer_sizes=(264,),alpha=0.1,activation='identity')
    ]

    best_scaler = [
        'RobustScaler','MinMaxScaler','MinMaxScaler','StandardScaler','None','StandardScaler','None','None'
    ]


    best_outlier_selection = [
        'YeoJonson','Replace','YeoJonson','Replace','YeoJonson','Replace','Log','Replace'
    ]

    best_test_size = [
        0.1,0.1,0.1,0.3,0.1,0.1,0.1,0.3
    ]

    return best_model, best_scaler, best_outlier_selection, best_test_size


