# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 13:29:51 2019

@author: Seba
"""

import pandas as pd
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.width', None)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

customers = pd.read_csv('Ecommerce Customers')

#sprawdzam czy występują brakujące dane
sns.heatmap(customers.isnull(), yticklabels = False, cbar = False,cmap='viridis')
print(customers.corr())

X = customers[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train , y_test = train_test_split(X,y, test_size=0.3, random_state = 101 )

from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train,y_train)
print(LR.intercept_)
print(LR.coef_)
cdf = pd.DataFrame(LR.coef_,X.columns, columns=['Coeff'] )
print(cdf)

predictions = LR.predict(X_test)

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test,predictions))
print('MSE:', metrics.mean_squared_error(y_test,predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test,predictions)))
print('EVS:', metrics.explained_variance_score(y_test,predictions)) 

plt.scatter(y_test,predictions)
sns.distplot((y_test-predictions),bins=50)