import pandas as pd
from DataAnalysis import *
from Preprocessing import *
from Model import *
from VisualizationOfResult import *
import warnings
warnings.filterwarnings("ignore")
#
# df = pd.read_csv('../data/train.csv')
#
#
# # plot_percentage_of_missing_values(df)
# #most_expensive_neighbourhoods(df)
# df = handle_missing_values(df)
# df, numerical_features = handle_outliers(df, method='Replace')
# df = transform_data(df)
#
# print(len(df.columns))
# #
# best_model, best_scaler, best_outlier_selection, best_test_size = best_models(i=8)
#
# # # plot_outliers(df, 'GrLivArea')



# #plot_outliers(df, 'GrLivArea')
# # para_cords(df)
#
# X_train, X_test, y_train, y_test = train_test_split_data(df, size=best_test_size)
# X_train, X_test = scale_data(X_train, X_test, numerical=numerical_features, scaler=best_scaler)

#
# best_model.fit(X_train,y_train)
# y_pred = best_model.predict(X_test)
# result = rmsle(y_test, y_pred)
# print(result)




#

# show_correlation_plot(df)
# plot_correlation(df)

# plot_outliers(df, 'LotArea')


#
# outler_handling = ['Replace','Log','YeoJonson']
# test_size = [0.1,0.2,0.3]
# scalers = ['None', 'StandardScaler', 'RobustScaler', 'MinMaxScaler']

#
# for outler in outler_handling:
#     df = pd.read_csv('../data/train.csv')
#     df = handle_missing_values(df)
#     df, numerical_features = handle_outliers(df, method=outler)
#     df = transform_data(df)
#     for testsize in test_size:
#         for scaler in scalers:
#             get_result_for_models(df, scaler, numerical_features, testsize, outler)



#FEATURE SELECTION

#feature_selection_selectKbest()


#feature_selection_RFE()

#PCA

# pca_reduction(True)

#Do the same in order to submit a result

# df = pd.read_csv('../data/train.csv')
# df = handle_missing_values(df)
# df, numerical_features = handle_outliers(df, method='Log')
# df = transform_data(df, alldummy=True)
# # # #
# test = pd.read_csv('../data/test.csv')
# index = list(test.Id)
# test = handle_missing_values(test, isTest=True)
# test, numerical_features1 = handle_outliers(test, method='Log', isTest = True)
# test = transform_data(test, alldummy=True)
# train_neural_network(test, df, numerical_features, 'MinMaxScaler', index)
# # plot_percentage_of_missing_values(test)
# create_submission_for_top3_models(test, df, numerical_features, 'MinMaxScaler', index)
#
# create_submission_for_bestfeatures(test, df, numerical_features, index)
#
#


#RMSE, MSE, MAE, R2
# other_measures()

#BestModels with dommy encoding for all
# for dummy in [True, False]:
#     all_dummy_encoding(alldummy=dummy)

# all_dummy_result()
# result_in_competition_alldummy()

# mean_fit_time()
# model_result_scaler()
# model_result_outlier()
# model_result_testsize()
# model_result_outlier_scaler('XGBRegressor')
# actual_result_in_competition()
# result_in_competition_feature_selection()
# best_model_kBest()
# best_model_RFE()
# best_model_PCA()


validation_result()