import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Function to delete columns
def removeCol(df, df2, colName):
    del df[colName]
    del df2[colName]
    return df, df2


# Function to impute missing values by region, in increasing order of suze
# Region -> LGA -> Ward -> Subvillage
def imputeLong(df):
    # Impute
    means_longitude_subvillage = df.groupby(['region', 'lga', 'ward', 'subvillage'])['longitude'].mean().reset_index()
    means_longitude_subvillage = means_longitude_subvillage.rename(columns={"longitude": "longitude_imputed_subvillage"})

    #ward level
    means_longitude_ward = df.groupby(['region', 'lga', 'ward',])['longitude'].mean().reset_index()
    means_longitude_ward = means_longitude_ward.rename(columns={"longitude": "longitude_imputed_ward"})

    #lga level
    means_longitude_lga = df.groupby(['region', 'lga'])['longitude'].mean().reset_index()
    means_longitude_lga = means_longitude_lga .rename(columns={"longitude": "longitude_imputed_lga"})

    # region_level
    means_longitude_region = df.groupby(['region'])['longitude'].mean().reset_index()
    means_longitude_region = means_longitude_region.rename(columns={"longitude": "longitude_imputed_region"})
    means_longitude_region.head()

    #merge the aggregated dataframes as new columns to the original df
    df= df.merge(means_longitude_subvillage, how = 'left', on = ['region', 'lga', 'ward', 'subvillage'])
    df= df.merge(means_longitude_ward, how = 'left', on = ['region', 'lga', 'ward'])
    df = df.merge(means_longitude_lga, how = 'left', on = ['region', 'lga'])
    df = df.merge(means_longitude_region, how = 'left', on = ['region'])

    #select the right longitude level based on the availability of information
    df['imputed_longitude'] = np.where(df['longitude'].isna(), df['longitude_imputed_subvillage'], df['longitude']) #if longitude is missing, impute it by the mean of the subvillage
    df['imputed_longitude'] = np.where(df['imputed_longitude'].isna(), df['longitude_imputed_ward'], df['imputed_longitude']) #if subvillage mean is missing, impute it by the ward
    df['imputed_longitude'] = np.where(df['imputed_longitude'].isna(), df['longitude_imputed_lga'], df['imputed_longitude'])
    df['imputed_longitude'] = np.where(df['imputed_longitude'].isna(), df['longitude_imputed_region'], df['imputed_longitude'])

    #drop redundant columns
    df= df.drop(['longitude_imputed_subvillage','longitude_imputed_ward' , 'longitude_imputed_lga' , 'longitude_imputed_region', 'longitude'], axis=1)
    return df


# Function to convert all strings to lower case
def toLowerCase(df):
    cols = [i for i in df.columns if type(df[i].iloc[0]) == str]
    # Convert all strings to lower case
    for col in cols:
        if df[col].dtype == 'object':
            df[col] = df[col].str.lower()
    return df


def catMrix(xtrain):
    #create a list of all categorical features
    categorical_cols = [cname for cname in xtrain.columns if
                        xtrain[cname].dtype == "object"]

    # Encode Categorical Columns 
    for col in categorical_cols:
        le = LabelEncoder()
        xtrain[col] = le.fit_transform(xtrain[col])

    xtrain = xtrain.drop('district_code',axis=1)

def corrMatrix(xtrain):
    # Create the correlation matrix
    corr_mean = xtrain.corr(method = 'pearson')

    #create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_mean, dtype=bool))

    # Add the mask to the heatmap
    fig, ax = plt.subplots(figsize=(25,25)) 
    ax = sns.heatmap(corr_mean, mask=mask, cmap= "RdYlGn", center=0, linewidths=1, annot=True, fmt=".2f")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_xticklabels())
    plt.show()

