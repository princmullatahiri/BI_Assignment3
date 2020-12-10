import pandas as pd
from scipy.stats import yeojohnson
import numpy as np
from scipy import stats

def transform_data(df):
    #ordinal_features = ['Alley','LotShape','LandContour','Utilities','LandSlope','ExterQual','ExterCond','BsmtQual','BsmtCond',
    #                    'BsmtExposure','BsmtFinType1','BsmtFinType2','HeatingQC','CentralAir','KitchenQual','Functional','FireplaceQu',
    #                    'GarageFinish','GarageQual','GarageCond','PavedDrive','PoolArea','PoolQC','Fence']

    #Turn categorical values to ordinal
    df = df.replace({"Alley" : {"Grvl":1, "Pave":2},
                     "LotShape": {"IR3":1,"IR2":2,"IR1":3,"Reg":4},
                     "LandContour": {"Low":1,"HLS":2,"Bnk":3,"Lvl":4},
                     "Utilities": {"ELO":1,"NoSeWa":2,"NoSewr":3,"AllPub":4},
                     "LandSlope": {"Sev":1,"Mod":2,"Gtl":3},
                     "ExterQual": {"Po":1,"Fa":2,"TA":3,"Gd":4,"Ex":5},
                     "ExterCond": {"Po":1,"Fa":2,"TA":3,"Gd":4,"Ex":5},
                     "BsmtQual": {"Po":1,"Fa":2,"TA":3,"Gd":4,"Ex":5},
                     "BsmtCond": {"Po":1,"Fa":2,"TA":3,"Gd":4,"Ex":5},
                     "BsmtExposure": {"No":1,"Mn":2,"Av":3,"Gd":4},
                     "BsmtFinType1": {"Unf":1,"LwQ":2,"Rec":3,"BLQ":4,"ALQ":5,"GLQ":6},
                     "BsmtFinType2": {"Unf":1,"LwQ":2,"Rec":3,"BLQ":4,"ALQ":5,"GLQ":6},
                     "HeatingQC": {"Po":1,"Fa":2,"TA":3,"Gd":4,"Ex":5},
                     "CentralAir": {"N":0,"Y":1},
                     "KitchenQual": {"Po":1,"Fa":2,"TA":3,"Gd":4,"Ex":5},
                     "Functional": {"Sal":1,"Sev":2,"Maj2":3,"Maj1":4,"Mod":5,"Min2":6,"Min1":7,"Typ":8},
                     "FireplaceQu": {"Po":1,"Fa":2,"TA":3,"Gd":4,"Ex":5},
                     "GarageFinish": {"Unf":1,"RFn":2,"Fin":3},
                     "GarageQual": {"Po":1,"Fa":2,"TA":3,"Gd":4,"Ex":5},
                     "GarageCond": {"Po":1,"Fa":2,"TA":3,"Gd":4,"Ex":5},
                     "PavedDrive": {"N":1,"P":2,"Y":3},
                     "PoolQC": {"Fa":1,"TA":2,"Gd":3,"Ex":4},
                     "Fence": {"MnWw":1,"GdWo":2,"MnPrv":3,"GdPrv":4},
                     "Street": {"Grvl":1,"Pave":2}
                    })

    df['BsmtCond'] = df['BsmtCond'].astype('int64')
    # Turn some numbers to real categorical values
    df = df.replace({"MSSubClass": {20:"MSSC20",30:"MSSC30",40:"MSSC40",45:"MSSC45",50:"MSSC50",60:"MSSC60",70:"MSSC70",75:"MSSC75",
                                    80:"MSSC80",85:"MSSC85",90:"MSSC90",120:"MSSC120",150:"MSSC150",160:"MSSC160",180:"MSSC180",190:"MSSC190"}
                     })

    # Convert categorical variable into dummy/indicator variables.
    dummy_features= ['MSSubClass','MSZoning','LotConfig','Neighborhood','Condition1','Condition2','BldgType','HouseStyle',
                     'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation','Heating','Electrical',
                     'GarageType','MiscFeature','SaleType','SaleCondition']
    df = pd.get_dummies(df,columns=dummy_features)
    df = df.set_index('Id')
    return df


def handle_missing_values(df, isTest=False):
    #These features the NAN represent when they dont have Alley,Fence, Pool etc.
    df.loc[:, "Alley"] = df.loc[:, "Alley"].fillna(0)
    df.loc[:, "MasVnrType"] = df.loc[:, "MasVnrType"].fillna("None")
    df.loc[:, "MasVnrArea"] = df.loc[:, "MasVnrArea"].fillna(0)
    df.loc[:, "BsmtQual"] = df.loc[:, "BsmtQual"].fillna(0)
    df.loc[:, "BsmtCond"] = df.loc[:, "BsmtCond"].fillna(0)
    df.loc[:, "BsmtExposure"] = df.loc[:, "BsmtExposure"].fillna(0)
    df.loc[:, "BsmtFinType1"] = df.loc[:, "BsmtFinType1"].fillna(0)
    df.loc[:, "BsmtFinType2"] = df.loc[:, "BsmtFinType2"].fillna(0)
    df.loc[:, "FireplaceQu"] = df.loc[:, "FireplaceQu"].fillna(0)
    df.loc[:, "GarageType"] = df.loc[:, "GarageType"].fillna(0)
    df.loc[:, "GarageYrBlt"] = df.loc[:, "GarageYrBlt"].fillna(0)
    df.loc[:, "GarageFinish"] = df.loc[:, "GarageFinish"].fillna(0)
    df.loc[:, "GarageQual"] = df.loc[:, "GarageQual"].fillna(0)
    df.loc[:, "GarageCond"] = df.loc[:, "GarageCond"].fillna(0)
    df.loc[:, "PoolQC"] = df.loc[:, "PoolQC"].fillna(0)
    df.loc[:, "Fence"] = df.loc[:, "Fence"].fillna(0)
    df.loc[:, "PoolQC"] = df.loc[:, "PoolQC"].fillna(0)
    df.loc[:, "MiscFeature"] = df.loc[:, "MiscFeature"].fillna("None")
    if isTest == True:
        df = df.fillna(0)
    # df.loc[:, "BsmtFinSF1"] = df.loc[:, "BsmtFinSF1"].fillna(0)
    #Since only one row has missing value for Electricity drop it:
    df = df[df.Id != 1380]

    #Drop column LotFrontage since it contains more than 15% missing values which whould affect our algorithm if we try to fill them
    df = df.drop('LotFrontage', axis = 1)

    return df


def handle_outliers(df, method="Log", isTest = False):
    #logtransform them
    numerical_features = df.select_dtypes(exclude=["object"]).columns
    other_featuers = df.select_dtypes(include=["object"]).columns
    df_numeric = df[numerical_features]
    df_other = df[other_featuers]
    not_numeric = ['Id','YearBuilt','YearRemodAdd','MoSold','YrSold', 'MSSubClass']
    if isTest == False:
        not_numeric.append('SalePrice')
        df_other = df_other.assign(SalePrice=df_numeric.loc[:, 'SalePrice'])


    df_other = df_other.assign(Id=df_numeric.loc[:,'Id'])
    df_other = df_other.assign(YearBuilt=df_numeric.loc[:, 'YearBuilt'])
    df_other = df_other.assign(YearRemodAdd=df_numeric.loc[:, 'YearRemodAdd'])
    df_other = df_other.assign(MoSold=df_numeric.loc[:, 'MoSold'])
    df_other = df_other.assign(YrSold=df_numeric.loc[:, 'YrSold'])

    df_other = df_other.assign(MSSubClass=df_numeric.loc[:, 'MSSubClass'])

    df_numeric = df_numeric.drop(not_numeric,axis=1)

    if method == "Log":
        #For normally distributed data, the skewness should be about zero. For unimodal continuous distributions, a skewness value greater than zero means that there is more weight in the right tail of the distribution.
        skewness = df_numeric.apply(lambda x: stats.skew(x))
        skewness = skewness[abs(skewness) > 0.5]
        #print(str(len(skewness.index)) + " features are skewed, where we need to handle outliers.")
        skewed_features = skewness.index
        df_numeric[skewed_features] = np.log1p(df_numeric[skewed_features])
        transformed_numeric_df = df_numeric
        transformed_numeric_df = transformed_numeric_df.reset_index()
    elif method == "YeoJonson":
        # Yeo-Jonson transformation
        data = {}
        for i in range(0,len(df_numeric.columns)):
            col = df_numeric.columns[i]
            yf, lmbd = yeojohnson(df_numeric[col])
            data[col] = yf
        transformed_numeric_df = pd.DataFrame(data)
        transformed_numeric_df = transformed_numeric_df.reset_index()

    elif method == "Replace":
        # Winsorizing (replace the outliers with other values)
        transformed_numeric_df = df_numeric.copy()
        for i in range(0, len(df_numeric.columns)):
            col = df_numeric.columns[i]
            lower_value = df[col].quantile(0.025)
            upper_value = df[col].quantile(0.975)

            df_sorted = df_numeric.sort_values([col], ascending=[False])
            values_to_replace = df_sorted[(df_sorted[col] <= upper_value) & (df_sorted[col] >= lower_value)]
            top5percent = round(len(values_to_replace) * 0.025)
            upper_values_to_replace = list(values_to_replace[col].head(top5percent))
            lower_values_to_replace = list(values_to_replace[col].tail(top5percent))
            transformed_numeric_df.loc[df_sorted[col] > upper_value, col] = df_numeric.loc[
                df[col] > upper_value, col].apply(
                lambda x: np.random.choice(upper_values_to_replace, 1, replace=True)[0])

            transformed_numeric_df.loc[df_sorted[col] < lower_value, col] = df_numeric.loc[
                df[col] < lower_value, col].apply(
                lambda x: np.random.choice(lower_values_to_replace, 1, replace=True)[0])

        transformed_numeric_df = transformed_numeric_df.reset_index()

    else:
        raise Exception("No such method: " + str(method) + ", please select form: Log, YeoJonson, Replace.")


    df_other = df_other.reset_index()
    df_other = df_other.drop('index', axis=1)
    transformed_numeric_df = transformed_numeric_df.drop('index', axis=1)
    df = pd.concat([transformed_numeric_df, df_other], axis=1)

    return df,list(df_numeric.columns)
