import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
data = pd.read_csv('insurance.csv')
# data.head()
# data.info()
# data['region'].value_counts().sort_values()
# data['children'].value_counts().sort_values()

clean_data = {'sex': {'male' : 0 , 'female' : 1} ,
                 'smoker': {'no': 0 , 'yes' : 1},
                   'region' : {'northwest':0, 'northeast':1,'southeast':2,'southwest':3}             }
data_copy = data.copy()
data_copy.replace(clean_data, inplace=True)

data_copy.describe()
data_pre = data_copy.copy()
tempBmi = data_pre.bmi
tempBmi = tempBmi.values.reshape(-1,1)
data_pre['bmi'] = StandardScaler().fit_transform(tempBmi)
tempAge = data_pre.age
tempAge = tempAge.values.reshape(-1,1)
data_pre['age'] = StandardScaler().fit_transform(tempAge)
tempCharges = data_pre.charges
tempCharges = tempCharges.values.reshape(-1,1)
data_pre['charges'] = StandardScaler().fit_transform(tempCharges)
# print(data_pre.head())

X = data_pre.drop('charges',axis=1).values
y = data_pre['charges'].values.reshape(-1,1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
# print('Size of X_train : ', X_train.shape)
# print('Size of y_train : ', y_train.shape)
# print('Size of X_test : ', X_test.shape)
# print('Size of Y_test : ', y_test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR


from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
cv_linear_reg = cross_val_score(estimator = linear_reg, X = X, y = y, cv = 10)
y_pred_linear_reg_train = linear_reg.predict(X_train)
r2_score_linear_reg_train = r2_score(y_train, y_pred_linear_reg_train)
y_pred_linear_reg_test = linear_reg.predict(X_test)
r2_score_linear_reg_test = r2_score(y_test, y_pred_linear_reg_test)
rmse_linear = (np.sqrt(mean_squared_error(y_test, y_pred_linear_reg_test)))
# print('CV Linear Regression : {0:.3f}'.format(cv_linear_reg.mean()))
# print('R2_score (train) : {0:.3f}'.format(r2_score_linear_reg_train))
# print('R2_score (test) : {0:.3f}'.format(r2_score_linear_reg_test))
# print('RMSE : {0:.3f}'.format(rmse_linear))

import pickle
pickle.dump(linear_reg,open("insurance.pkl","wb"))

