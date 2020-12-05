import pandas as pd
import matplotlib.pyplot as plt

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
    plt.axhline(y=20,color='red',ls=':')
    plt.title('Columns with Missing Values')
    plt.xlabel('Missing Values')
    plt.ylabel('Percentage of Missing Values')
    plt.show()

