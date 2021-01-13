import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import json

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
    #plt.savefig('../visualizations/Data Analysis/missing_values_percentage.png')
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
    plt.savefig('../visualizations/Data Analysis/correlation_plot.png')
    plt.show()

def plot_outliers(df,column):
    sns.boxplot(y=df[column])
    plt.title("Boxplot for showing Outliers for: " + str(column))
    plt.savefig('../visualizations/Data Analysis/boxplot_'+column+'.png')
    plt.show()

def plot_pca(x_test,y_pred_original, model):

    scaler = MinMaxScaler(feature_range=(1,60))
    y_pred = scaler.fit_transform(y_pred_original.reshape(-1, 1))
    y_pred = pd.Series(y_pred[:, 0])
    y_pred = np.round(y_pred,2)
    y_pred_original = np.round(y_pred_original,0)


    x_test_numpy = np.array(x_test)
    transposed = x_test_numpy.T
    transpose_list = transposed.tolist()
    y_pred_list = list(y_pred)
    pca1_list = transpose_list[0]
    pca2_list = transpose_list[1]
    data = []

    main_graph = go.Scatter(
        x=pca1_list,
        y=pca2_list,
        mode='markers',
        marker=dict(size=y_pred_list, color='rgb(31, 119, 180)'),
        text=list(y_pred_original),
        showlegend=False,
        hovertemplate=
        '<b>SoldPrice:</b> %{text}' +
        '<br><b>PC1:</b> %{x:.2f}<br>' +
        '<b>PC2:</b> %{y:.2f}<extra></extra>',
    )


    data.append(main_graph)

    y_pred_original = np.sort(y_pred_original)[::-1]
    y_pred_original = list(y_pred_original)

    y_pred_legend = np.sort(y_pred)[::-1]
    y_pred_legend = list(y_pred_legend)

    legend_size = 12
    bins_y_pred = pd.cut(y_pred_original, bins=legend_size).categories
    bins_y_scaled = pd.cut(y_pred_legend, bins=legend_size).categories




    for i in range(legend_size- 1, -1, -1):
        lgngroup = str(bins_y_pred[i].left) + '-' + str(bins_y_pred[i].right)
        data.append(go.Scatter(x=[None], y=[None], mode='markers',
                               marker=dict(size=bins_y_scaled[i].left,color='rgb(31, 119, 180)',sizeref=0.1),
                               legendgroup=lgngroup, showlegend=True,
                               hovertemplate='<extra></extra>',
                               name=lgngroup),
                               )

    fig = go.Figure(data=data)
    fig.update_layout(title='<b>Visualization of PCA and predicted value of Sale Price for '+model+'</b>', xaxis_title="<b>PC1</b>",
                          yaxis_title="<b>PC2</b>", legend_title_text='<b>Sold Price/Size of Markers</b>', showlegend=True,
                      )
    fig.show()

def show_correlation_plot(df):
    corr = df.corr()
    corr.sort_values(["SalePrice"], ascending=False, inplace=True)
    dt = corr.SalePrice
    dt = dt[abs(dt) > 0.25]
    cols = list(dt.index)
    df = df[cols]
    corr=df.corr()
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(20, 230, as_cmap=True)
    sns.heatmap(corr, cmap=cmap)
    plt.savefig('../visualizations/Data Analysis/correlation_matrix_plot.png')
    plt.show()

def para_cords(df):
    corr = df.corr()
    corr.sort_values(["SalePrice"], ascending=False, inplace=True)
    dt = corr.SalePrice
    dt = dt[abs(dt) > 0.35]
    cols = list(dt.index)
    cols_to_get = [cols[0],cols[1],cols[2],cols[len(cols)-2],cols[len(cols)-1]]
    df = df[cols_to_get]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    scaled_df = pd.DataFrame(scaled_data, columns=cols_to_get)


    fig = px.parallel_coordinates(scaled_df, color="SalePrice", labels={"SalePrice": "Sale Price",
                                                                  "OverallQual": "Overwall Quality",
                                                                  "GrLivArea": "Living area (feet)",
                                                                  "GarageType_Detchd": "Detached Garage",
                                                                  "MasVnrType_None": "No Masonry Veneer", },
                                  color_continuous_scale=px.colors.diverging.Tealrose,
                                  color_continuous_midpoint=0.5)

    # fig.update_layout(title='<b>Correlation of Most Correlated Features</b>')
    fig.show()


def most_expensive_neighbourhoods(df):
    df = df[['Neighborhood','SalePrice']]


    df = df.groupby(['Neighborhood'], sort=False, as_index=False).mean()

    with open('../data/ames_geojson.geojson') as f:
        gj = json.load(f)

    fig = px.choropleth(df, geojson=gj, color="SalePrice",
                        locations="Neighborhood", featureidkey="properties.name",
                        color_continuous_scale='Blues',#["red", "yellow", "blue"],
                        projection="mercator"
                        )
    fig.update_geos(fitbounds="locations", visible=False)
    #fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_traces(marker_line_width=1.2, marker_line_color='black')
    fig.update_layout(title='<b>House Prices by Neighborhoods</b>')
    fig.show()


def statistical_plots(df):
    sns.displot(df, x="SalePrice")
    plt.xticks(rotation=45)
    plt.vlines(x=df.SalePrice.min(), color= 'red',linestyle='--',ymin=-0, ymax=178)
    plt.vlines(x=df.SalePrice.max(), color= 'red',linestyle='--',ymin=-0, ymax=140)
    plt.vlines(x=df.SalePrice.mean(), color= 'green',linestyle='--',ymin=-0, ymax=172)
    plt.vlines(x=df.SalePrice.median(), color= 'orange', linestyle='--',ymin=-0, ymax=178)

    plt.ylim(0, 190)
    minmax = mpatches.Patch(color='red', label='Min/Max')
    avg = mpatches.Patch(color='green', label='Average')
    med = mpatches.Patch(color='orange', label='Median')

    plt.legend(handles=[minmax, avg, med])

    plt.text(df.SalePrice.min(), 180, str(df.SalePrice.min()), color='red')
    plt.text(df.SalePrice.max(), 142, str(df.SalePrice.max()), color='red')
    plt.text(df.SalePrice.mean(), 174, str(df.SalePrice.mean()), color='green')
    plt.text(df.SalePrice.median(), 180, str(df.SalePrice.median()), color='orange')

    plt.title('Distribution of SalePrice')

    plt.show()