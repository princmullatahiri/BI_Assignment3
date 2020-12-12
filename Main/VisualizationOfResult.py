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
    plt.show()