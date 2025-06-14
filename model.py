import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')

def clean_data(df) -> list:
    to_remove = []
    for col in df.columns:
        if df[col].nunique() == 1:
            to_remove.append(col)
        
        elif (df[col].isnull()).mean() > 0.60:
            to_remove.append(col)
    
    return to_remove

if __name__ == "__main__":
    df = pd.read_csv('Zillow.csv')
    to_remove = clean_data(df)
    df.drop(to_remove, axis=1, inplace=True)
    print(df.info())