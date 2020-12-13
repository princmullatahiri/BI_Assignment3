import pandas as pd
from DataAnalysis import *
from Preprocessing import *
from Model import *
from VisualizationOfResult import *
import warnings
warnings.filterwarnings("ignore")
#
df = pd.read_csv('../data/train.csv')
most_expensive_neighbourhoods(df)
# df = handle_missing_values(df)
# # plot_outliers(df, 'GrLivArea')
# df, numerical_features = handle_outliers(df, method='Replace')
# df = transform_data(df)
# #plot_outliers(df, 'GrLivArea')
# para_cords(df)




#

#show_correlation_plot(df)
# plot_correlation(df)

# plot_outliers(df, 'LotArea')



outler_handling = ['Replace','Log','YeoJonson']
test_size = [0.1,0.2,0.3]
scalers = ['None', 'StandardScaler', 'RobustScaler', 'MinMaxScaler']

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
# df, numerical_features = handle_outliers(df, method='YeoJonson')
# df = transform_data(df)
#
# test = pd.read_csv('../data/test.csv')
# index = list(test.Id)
# test = handle_missing_values(test, isTest=True)
# test, numerical_features1 = handle_outliers(test, method='YeoJonson', isTest = True)
# test = transform_data(test)
# # create_submission_for_top3_models(test, df, numerical_features, 'MinMaxScaler', index)
#
# create_submission_for_bestfeatures(test, df, numerical_features, index)
#
#

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