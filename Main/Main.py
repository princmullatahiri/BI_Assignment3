import pandas as pd
from DataAnalysis import *
from Preprocessing import *
from Model import *


# plot_percentage_of_missing_values(df)
# plot_percentage_of_missing_values(df)
# plot_outliers(df, 'LotArea')
#Method = ["Log", "YeoJonson", "Replace"]
# plot_outliers(df, 'LotArea')



outler_handling = [  'Replace'] #'Log','YeoJonson',
test_size = [0.1,0.2,0.3]
scalers = ['None', 'StandardScaler', 'RobustScaler', 'MinMaxScaler']


# for outler in outler_handling:
#     df = pd.read_csv('../data/train.csv')
#     df = handle_missing_values(df)
#     df, numerical_features = handle_outliers(df, method=outler)
#     df = transform_data(df)
#     for testsize in test_size:
#         for scaler in scalers:
#             get_result_for_models(df, scaler, numerical_features, testsize, outler)
#

#TODO: PCA, Feature Selection

#FEATURE SELECTION

#feature_selection_selectKbest()


# feature_selection_RFE()

#Do the same in order to submit a result

df = pd.read_csv('../data/train.csv')
df = handle_missing_values(df)
df, numerical_features = handle_outliers(df, method='Replace')
df = transform_data(df)

test = pd.read_csv('../data/test.csv')
index = list(test.Id)
test = handle_missing_values(test, isTest=True)
test, numerical_features1 = handle_outliers(test, method='Replace', isTest = True)
test = transform_data(test)
#create_submission_for_top3_models(test, df, numerical_features, 'None', index)
create_submission_for_bestfeatures(test, df, numerical_features, index)