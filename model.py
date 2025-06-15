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
    plt.show()

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        elif df[col].dtype == np.number:
            df[col] = df[col].fillna(df[col].mean())
    
    print(df.isnull().sum().sum())

    ints, objects, floats = [], [], []

    # Exploratory Data Analysis
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
    
    plt.figure(figsize=(8, 5))
    sb.distplot(df['target'])
    plt.show()