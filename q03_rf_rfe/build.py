# Default imports
import pandas as pd
import numpy as np
data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


# Your solution code here
def rf_rfe(df):
    X,y=data.iloc[:,:-1],data.iloc[:,-1]
    model=RandomForestClassifier()
    rfe = RFE(model,17 )
    rfe = rfe.fit(X, y)
    df1=pd.DataFrame(np.vstack((X.columns,rfe.ranking_)))
    df1.columns=df1.iloc[1]
    df1= df1.T
    df1.columns=[['Features','Significant']]
    return list(df1[df1.Significant<=1].Features)
