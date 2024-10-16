import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data =  './Wholesale customers data.csv'
df = pd.read_csv(data)

def show_full_df_info():
    print(df.describe(), df.info())
    print(df.shape)
    print(df.isnull().sum())

show_full_df_info()

y = df['Channel']
X = df.drop('Channel', axis = 1)

#checking
#print(y.head())
#print(X.head())

y -= 1

d_matrix = xgb.DMatrix(data=X,label=y)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=0)

#parameters for classifier
params = {'objective':'binary:logistic',
            'max_depth': 6,
            'alpha': 10,
            'learning_rate': 1.0,
            'n_estimators':100}

xgb_classifier = XGBClassifier(**params)
xgb_classifier.fit(X_train,y_train)

#checking
#print(xgb_classifier)

y_pred = xgb_classifier.predict(X_test)
print('Accuracy:', format(accuracy_score(y_test, y_pred)))
print(y_test[y_test != y_pred])

xgb_classifier.save_model('model.json')
