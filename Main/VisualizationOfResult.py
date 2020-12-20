import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import numpy as np




def mean_fit_time():
    df_models = pd.read_csv('../Report/Model/Model Performances.csv')

    list_of_models = list(df_models.Model.unique())

    model = []
    mean_fit_time = []
    for i in range(0,len(list_of_models)):
        mdl = list_of_models[i]
        df = df_models[df_models.Model == mdl]
        mean = df['Fit Time(s)'].mean()
        model.append(mdl)
        mean_fit_time.append(mean)



    plt.bar(model, mean_fit_time)
    plt.xticks(rotation='vertical')
    plt.title('Model Average Fit Time with 10-Fold CV')
    plt.xlabel('Model')
    plt.ylabel('Fit Time (s)')
    plt.savefig('../visualizations/Results/score_models_fit_time.png')
    plt.show()


def model_result_scaler():
    df_models = pd.read_csv('../Report/Model/Model Performances.csv')

    list_of_models = list(df_models.Model.unique())
    scalers = list(df_models.Scaler.unique())

    barWidth = 0.15
    r1 = np.arange(len(list_of_models))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]
    r5 = [x + barWidth for x in r4]

    x_val = [r1,r2,r3,r4,r5]



    fig, ax = plt.subplots()


    for i in range(0,len(scalers)):
        scl = scalers[i]
        mean_result = []
        for j in range(0, len(list_of_models)):
            mdl = list_of_models[j]
            df = df_models[(df_models.Model == mdl) & (df_models.Scaler == scl)]
            mean = df['Test RMSLE'].mean()
            mean_result.append(mean)
        ax.bar(x_val[i], mean_result, barWidth, label=scl)

    plt.xticks([r + barWidth for r in range(len(list_of_models))], list_of_models)

    plt.xticks(rotation='vertical')
    plt.title('Model Average Score for different Scalers')

    plt.xlabel('Model')
    plt.ylabel('RMSLE')
    ax.legend()
    plt.savefig('../visualizations/Results/score_models_scalers.png')
    plt.show()



def model_result_outlier():
    df_models = pd.read_csv('../Report/Model/Model Performances.csv')

    list_of_models = list(df_models.Model.unique())
    outliers = list(df_models['Outlier Handling'].unique())

    barWidth = 0.20
    r1 = np.arange(len(list_of_models))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    x_val = [r1,r2,r3]



    fig, ax = plt.subplots()


    for i in range(0,len(outliers)):
        out = outliers[i]
        mean_result = []
        for j in range(0, len(list_of_models)):
            mdl = list_of_models[j]
            df = df_models[(df_models.Model == mdl) & (df_models['Outlier Handling'] == out)]
            mean = df['Test RMSLE'].mean()
            mean_result.append(mean)
        ax.bar(x_val[i], mean_result, barWidth, label=out)

    plt.xticks([r + barWidth for r in range(len(list_of_models))], list_of_models)

    plt.xticks(rotation='vertical')
    plt.title('Model Average Score for different Outlier Handlers')

    plt.xlabel('Model')
    plt.ylabel('RMSLE')
    ax.legend()
    plt.savefig('../visualizations/Results/score_model_outlier_handling.png')
    plt.show()


def model_result_testsize():
    df_models = pd.read_csv('../Report/Model/Model Performances.csv')

    list_of_models = list(df_models.Model.unique())
    testsize = list(df_models['Test Size'].unique())

    barWidth = 0.20
    r1 = np.arange(len(list_of_models))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    x_val = [r1,r2,r3]



    fig, ax = plt.subplots()


    for i in range(0,len(testsize)):
        tsts = testsize[i]
        mean_result = []
        for j in range(0, len(list_of_models)):
            mdl = list_of_models[j]
            df = df_models[(df_models.Model == mdl) & (df_models['Test Size'] == tsts)]
            mean = df['Test RMSLE'].mean()
            mean_result.append(mean)
        ax.bar(x_val[i], mean_result, barWidth, label=tsts)

    plt.xticks([r + barWidth for r in range(len(list_of_models))], list_of_models)

    plt.xticks(rotation='vertical')
    plt.title('Model Average Score for different Test Sizes')

    plt.xlabel('Model')
    plt.ylabel('RMSLE')
    ax.legend()
    plt.savefig('../visualizations/Results/score_models_test_size.png')
    plt.show()


def model_result_outlier_scaler(model):
    df_models = pd.read_csv('../Report/Model/Model Performances.csv')
    df_models = df_models[df_models.Model == model]

    list_of_scalers = list(df_models.Scaler.unique())
    outliers = list(df_models['Outlier Handling'].unique())

    barWidth = 0.20
    r1 = np.arange(len(list_of_scalers))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    x_val = [r1,r2,r3]



    fig, ax = plt.subplots()


    for i in range(0,len(outliers)):
        out = outliers[i]
        mean_result = []
        for j in range(0, len(list_of_scalers)):
            scl = list_of_scalers[j]
            df = df_models[(df_models.Scaler == scl) & (df_models['Outlier Handling'] == out)]
            mean = df['Test RMSLE'].mean()
            mean_result.append(mean)
        ax.bar(x_val[i], mean_result, barWidth, label=out)

    plt.xticks([r + barWidth for r in range(len(list_of_scalers))], list_of_scalers)

    plt.xticks(rotation='vertical')
    plt.title('Model Average Score for '+ model)

    plt.xlabel('Scaler')
    plt.ylabel('RMSLE')
    ax.legend()
    plt.savefig('../visualizations/Results/score_'+model+'_outlier_scaler.png')
    plt.show()


def best_model_kBest():
    df_models = pd.read_csv('../Report/FeatureSelection/FeatureSelection_SelectKBest.csv')

    list_of_models = list(df_models.Model.unique())
    list_of_models.remove('MLPRegressor')
    topatt = [
        [1,10],[15,25],[30,40],[50,70],[80,100]
    ]

    barWidth = 0.1
    r1 = np.arange(len(list_of_models))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]
    r5 = [x + barWidth for x in r4]

    x_val = [r1, r2, r3, r4, r5]

    fig, ax = plt.subplots()

    for i in range(0, len(topatt)):
        top = topatt[i]
        mean_result = []
        for j in range(0, len(list_of_models)):
            mdl = list_of_models[j]
            df = df_models[(df_models.Model == mdl) & (df_models['Top Attributes'] >= top[0]) & (df_models['Top Attributes'] <= top[1])]
            mean = df['Test RMSLE'].mean()
            mean_result.append(mean)
        topstr = str(top[0])+'-'+str(top[1])
        ax.bar(x_val[i], mean_result, barWidth, label=topstr)

    plt.xticks([r + barWidth for r in range(len(list_of_models))], list_of_models)

    plt.xticks(rotation='vertical')
    plt.title('Model Average Score for # of Feature Selection(kBest)')

    plt.xlabel('Model')
    plt.ylabel('RMSLE')
    ax.legend(bbox_to_anchor=(1.05, -0.16))
    plt.savefig('../visualizations/Results/score_models_kBest.png')
    plt.show()


def best_model_RFE():
    df_models = pd.read_csv('../Report/FeatureSelection/FeatureSelection_RFE.csv')
    list_of_models = list(df_models.Model.unique())
    list_of_models.remove('LinearRegression')
    topatt = [
        [1,10],[15,25],[30,40],[50,70]
    ]

    barWidth = 0.1
    r1 = np.arange(len(list_of_models))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]

    x_val = [r1, r2, r3, r4]

    fig, ax = plt.subplots()

    for i in range(0, len(topatt)):
        top = topatt[i]
        mean_result = []
        for j in range(0, len(list_of_models)):
            mdl = list_of_models[j]
            df = df_models[(df_models.Model == mdl) & (df_models['Top Attributes'] >= top[0]) & (df_models['Top Attributes'] <= top[1])]
            mean = df['Test RMSLE'].mean()
            mean_result.append(mean)
        topstr = str(top[0])+'-'+str(top[1])
        ax.bar(x_val[i], mean_result, barWidth, label=topstr)

    plt.xticks([r + barWidth for r in range(len(list_of_models))], list_of_models)

    plt.xticks(rotation='vertical')
    plt.title('Model Average Score for # of Feature Selection(RFE)')

    plt.xlabel('Model')
    plt.ylabel('RMSLE')
    ax.legend()
    plt.savefig('../visualizations/Results/score_modelsRFE.png')
    plt.show()

def best_model_PCA():
    df_models = pd.read_csv('../Report/FeatureSelection/DimensionalityReduction_PCA.csv')

    list_of_models = list(df_models.Model.unique())
    list_of_models.remove('MLPRegressor')
    topatt = [
        [2,10],[20,30],[40,60],[70,100]
    ]

    barWidth = 0.1
    r1 = np.arange(len(list_of_models))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]

    x_val = [r1, r2, r3, r4]

    fig, ax = plt.subplots()

    for i in range(0, len(topatt)):
        top = topatt[i]
        mean_result = []
        for j in range(0, len(list_of_models)):
            mdl = list_of_models[j]
            df = df_models[(df_models.Model == mdl) & (df_models['Nr. of Components'] >= top[0]) & (df_models['Nr. of Components'] <= top[1])]
            mean = df['Test RMSLE'].mean()
            mean_result.append(mean)
        topstr = str(top[0])+'-'+str(top[1])
        ax.bar(x_val[i], mean_result, barWidth, label=topstr)

    plt.xticks([r + barWidth for r in range(len(list_of_models))], list_of_models)

    plt.xticks(rotation='vertical')
    plt.title('Model Average Score for # of Components(PCA)')

    plt.xlabel('Model')
    plt.ylabel('RMSLE')
    ax.legend(bbox_to_anchor=(1.05, -0.16))
    plt.savefig('../visualizations/Results/score_models_PCA.png')
    plt.show()


def actual_result_in_competition():
    list_of_models = ['AdaBoostRegressor','GradientBoostingRegressor','KNeighnorsRegressor','RandomForestRegressor','ElasticNet','XGBRegressor','KerasNeuralNetwork']
    result_in_test = [0.1529,0.1381,0.1696,0.1366,0.1342, 0.12809, 0.1571]
    result_in_competition = [0.15956,0.39244,0.17773, 0.23571, 0.15423, 0.1472, 0.20947]

    barWidth = 0.35
    r1 = np.arange(len(list_of_models))
    r2 = [x + barWidth for x in r1]



    fig, ax = plt.subplots()


    ax.bar(r1, result_in_test, barWidth, label='Test Result')
    ax.bar(r2, result_in_competition, barWidth, label='Competition Result')

    plt.xticks([r + barWidth for r in range(len(list_of_models))], list_of_models)

    plt.xticks(rotation='vertical')
    plt.title('Model Score in Competition')

    plt.xlabel('Model')
    plt.ylabel('RMSLE')
    ax.legend()
    plt.savefig('../visualizations/Results/score_competition.png')
    plt.show()



def result_in_competition_feature_selection():
    list_of_models = ['ElasticNet/70','GradientBoostingRegressor/25','XGBRegressor/50','RandomForestRegressor/50']
    result_with_feature_selection = [0.15500,0.34636,0.14371,0.24375]
    result_in_competition = [0.15423,0.39244,0.14720,0.23571]

    barWidth = 0.35
    r1 = np.arange(len(list_of_models))
    r2 = [x + barWidth for x in r1]



    fig, ax = plt.subplots()


    ax.bar(r1, result_with_feature_selection, barWidth, label='With Feature Selection')
    ax.bar(r2, result_in_competition, barWidth, label='Without Feature Selection')

    plt.xticks([r + barWidth for r in range(len(list_of_models))], list_of_models)

    plt.xticks(rotation='vertical')
    plt.title('Model Score in Competition with/without Feature Selection (RFE)')

    plt.xlabel('Model')
    plt.ylabel('RMSLE')
    ax.legend()
    plt.savefig('../visualizations/Results/score_competition_feature_selection.png')
    plt.show()


def plot_curve_neural_network(history):
    plt.title('Learning Curves')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    plt.savefig('../visualizations/Results/curve_neural_network.png')
    plt.show()

def all_dummy_result():
    df = pd.read_csv('../Report/Model/TransformationPerformances.csv')
    df = df.sort_values('Model')


    models = df.Model.unique()

    barWidth = 0.35
    r1 = np.arange(len(models))
    r2 = [x + barWidth for x in r1]


    fig, ax = plt.subplots()



    ax.bar(r1, list(df[df.allDummy == True].RMSLE), barWidth, label='All Dummy')
    ax.bar(r2, list(df[df.allDummy == False].RMSLE), barWidth, label='Ordinal Values')

    plt.xticks([r + barWidth for r in range(len(models))], models)

    plt.xticks(rotation='vertical')
    plt.title('Model Scores')

    plt.xlabel('Model')
    plt.ylabel('RMSLE')
    ax.legend()
    #plt.savefig('../visualizations/Results/score_competition.png')
    plt.show()

def result_in_competition_alldummy():
    list_of_models = ['ElasticNet','XGBRegressor','RandomForestRegressor']
    result = [0.15423,0.14720,0.23571]
    result_with_alldummy = [0.14226,0.14059,0.24572]

    barWidth = 0.35
    r1 = np.arange(len(list_of_models))
    r2 = [x + barWidth for x in r1]



    fig, ax = plt.subplots()


    ax.bar(r1, result, barWidth, label='Ordinal Values')
    ax.bar(r2, result_with_alldummy, barWidth, label='All Dummy')

    plt.xticks([r + barWidth for r in range(len(list_of_models))], list_of_models)

    plt.xticks(rotation='vertical')
    plt.title('Model Score in Competition with/without All Dummys')

    plt.xlabel('Model')
    plt.ylabel('RMSLE')
    ax.legend()
    # plt.savefig('../visualizations/Results/score_competition_feature_selection.png')
    plt.show()

