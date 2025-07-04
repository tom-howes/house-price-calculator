import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae

import warnings
warnings.filterwarnings('ignore')

def remove_null(df):
    to_remove = []
    for col in df.columns:
        if df[col].nunique() == 1:
            to_remove.append(col)
        
        elif (df[col].isnull()).mean() > 0.60:
            to_remove.append(col)
    df.drop(to_remove, axis=1, inplace=True)
    return df

if __name__ == "__main__":
    df = pd.read_csv('Zillow.csv')
    df = remove_null(df)
    print(df.info())
    df.isnull().sum().plot.bar()
    # plt.show()

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        elif df[col].dtype == np.number:
            df[col] = df[col].fillna(df[col].mean())
    
    print(df.isnull().sum().sum())

    ints, objects, floats = [], [], []

### Exploratory Data Analysis ###
    for col in df.columns:
        if df[col].dtype == float:
            floats.append(col)
        elif df[col].dtype == int:
            ints.append(col)
        else:
            objects.append(col)
    
    print(len(ints), len(floats), len(objects))

    for col in objects:
        print(col, '->', df[col].nunique())
        print(df[col].unique())
        print()
    
    # # Distribution plot
    # plt.figure(figsize=(8, 5))
    # sb.distplot(df['target'])
    # plt.show()

    # # Box plot to detect outliers
    # plt.figure(figsize=(8, 5))
    # sb.boxplot(df['target'])
    # plt.show()

    print('Shape of the dataframe before removal of outliers', df.shape)
    df = df[(df['target'] > -1) & (df['target']< 1)]
    print('Shape of the dataframe after remove of outliers', df.shape)

    for col in objects:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # # Check highly correlated features
    # plt.figure(figsize=(15, 15))
    # sb.heatmap(df.corr() > 0.8,
    #            annot=True,
    #            cbar=False)
    # plt.show()

    to_remove = ['calculatedbathnbr', 'fullbathcnt', 'fips',
                 'rawcensustractandblock', 'taxvaluedollarcnt',
                 'finishedsquarefeet12', 'landtaxvaluedollarcnt']
    df.drop(to_remove, axis=1, inplace=True)
    
### Model Training ###

    df.dropna(inplace=True)

    features = df.drop(['parcelid'], axis=1)
    target = df['target'].values

    # Split into training and testing data

    X_train, X_val, \
        Y_train, Y_val = train_test_split(features, target,
                                          test_size=0.1,
                                          random_state=22)
    print(X_train.shape, X_val.shape)

    # Normalizing features for stable and faster training

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    models = [LinearRegression(), XGBRegressor(), Lasso(), RandomForestRegressor(), Ridge()]

    for i in range(5):
        models[i].fit(X_train, Y_train)
        print(f"{models[i]} : ")

        train_preds = models[i].predict(X_train)
        print('Training Error : ', mae(Y_train, train_preds))

        val_preds = models[i].predict(X_val)
        print('Validation Error : ', mae(Y_val, val_preds))
        print()