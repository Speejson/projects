# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 22:16:36 2019

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

ad_data = pd.read_csv('advertising.csv')
ad_data.info()
ad_data.describe()

ad_data['Age'].plot.hist(bins=30)

sns.jointplot(y='Area Income', x='Age', data= ad_data)

sns.jointplot(y='Daily Time Spent on Site', x='Age', data= ad_data, kind='kde', color='r')

sns.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage', data= ad_data)

sns.pairplot( data=ad_data, hue='Clicked on Ad')

#logistic regression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X,y, test_size = 0.3, random_state= 101)
X = ad_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']]
y = ad_data['Clicked on Ad']
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test, predictions))