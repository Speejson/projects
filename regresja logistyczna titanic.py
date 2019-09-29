# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 22:38:53 2019

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

titanic = pd.read_csv('titanic_train.csv')

sns.heatmap(titanic.isnull(), yticklabels = False, cbar = False,cmap='viridis')

nulls = pd.DataFrame(titanic.dtypes, columns=['dtype'])
nulls['Sum_of_Nulls']= pd.DataFrame(titanic.isnull().sum())
nulls['Per_of_Nulls'] = round((titanic.apply(pd.isnull).mean()*100),2)
print(nulls)

titanic.drop('Cabin', axis = 1, inplace= True)
print(round(titanic[['Age','Pclass']].groupby('Pclass').mean(),0))
P1 = 38
P2 = 30
P3 = 25

#funkcja uzupełniająca brakujące dane wieku rednia wieku pasażerów w danej klasie
def age_fill(columns):
    Age = columns[0]
    Pclass = columns[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return P1
        elif Pclass == 2:
            return P2
        elif Pclass == 3:
            return  P3
    else: 
        return Age

titanic['Age'] = titanic[['Age','Pclass']].apply(age_fill, axis=1)

sex = pd.get_dummies(titanic['Sex'],drop_first=True)
embark = pd.get_dummies(titanic['Embarked'],drop_first = True)
titanic = pd.concat([titanic,sex,embark],axis =1)

titanic.drop(['Name','Embarked','Sex','Ticket'],axis =1, inplace=True)
titanic.drop(['PassengerId'],axis =1, inplace=True)

X = titanic[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q',
       'S']]
y = titanic['Survived']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)

predictions = model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test, predictions))



        
    