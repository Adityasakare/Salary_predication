# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 19:38:59 2019

@author: Aditya
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values 
y = dataset.iloc[:, 1].values 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)
y_pred = reg.predict(X_train)

plt.scatter(X_train,y_train,color="blue")
plt.plot(X_train,reg.predict(X_train),color="red")
plt.title("Training Data")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

plt.scatter(X_test,y_test,color="blue")
plt.plot(X_train,reg.predict(X_train),color="red")
plt.title("Test Data")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

from sklearn.model_selection import cross_val_score
cross_val_score(regressor, X_train, y_train,cv=10).mean()
