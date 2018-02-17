# Default imports

import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression


# Write your solution here:
def percentile_k_features(df,k=20):
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    #listn=SelectPercentile(f_regression,k).fit_transform(X,y)
    listn,_=f_regression(X,y)
    df1=pd.DataFrame(np.vstack((X.columns,listn)))
    df2 =df1.rename(columns=df1.iloc[0]).reindex(df1.index.drop(0)).T
    df2.columns = [['imp']]
    df2[['imp']] = df2[['imp']].apply(pd.to_numeric)
    return  list(df2.nlargest(10,'imp').index.values)
    
