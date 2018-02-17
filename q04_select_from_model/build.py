# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')


# Your solution code here
def select_from_model(df):
    X,y=data.iloc[:,:-1],data.iloc[:,-1]

    clf=RandomForestClassifier()
    clf.fit(X, y)
    model=SelectFromModel(clf, prefit=True)
    df1= pd.DataFrame(np.vstack((X.columns.T,model.get_support())))
    #return type(list(df1[df1.Significant<=1].Features))
    df1.columns=df1.iloc[0]
    df1= df1.T
    df1.columns=[['Features','Significance']]
    return list(df1[df1.Significance==True]['Features'])
