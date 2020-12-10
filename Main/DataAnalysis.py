import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_percentage_of_missing_values(df):
    missing_values = df.isnull().sum()
    length = len(df)/100
    missing_values = round(missing_values/length,2)
    missing_val_dict = {}
    for i in range(0,len(missing_values)):
        if missing_values[i] > 0.0:
            missing_val_dict[missing_values.index[i]] = missing_values[i]

    plt.bar(list(missing_val_dict.keys()), list(missing_val_dict.values()))
    plt.xticks(rotation='vertical')
    plt.axhline(y=15,color='red',ls=':')
    plt.title('Columns with Missing Values')
    plt.xlabel('Missing Values')
    plt.ylabel('Percentage of Missing Values')
    plt.show()
    return missing_val_dict

def plot_correlation(df):
    corr = df.corr()
    corr.sort_values(["SalePrice"], ascending=False, inplace=True)
    dt = corr.SalePrice
    dt = dt.drop(['SalePrice'])
    dt = dt[abs(dt) > 0.3]
    sns.barplot(y=list(dt.index), x=list(dt.values), orient='h', palette='viridis')
    sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True)
    plt.title('Most Correlated Features')
    plt.show()

def plot_outliers(df,column):
    sns.boxplot(y=df[column])
    plt.title("Boxplot for showing Outliers for: " + str(column))
    plt.show()