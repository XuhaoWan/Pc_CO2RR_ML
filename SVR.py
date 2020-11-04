#!/usr/bin/env python
# -*- coding:utf-8 -*-
#author: xhwan

import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


data_train = np.loadtxt('data_train.csv', delimiter=",", dtype="float")
x = data_train[..., 0:20 ]
ss = MinMaxScaler()
x = ss.fit_transform(x)
y = data_train[...,20 ]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=16)


model = SVR(kernel='rbf')
model.fit(x_train, y_train)

rmse = np.sqrt(mse(y_train,model.predict(x_train)))
r2 = r2_score(y_train,model.predict(x_train))
rmset = np.sqrt(mse(y_test,model.predict(x_test)))
r2t = r2_score(y_test,model.predict(x_test))
print(model.predict(x_test))
print(y_test)
print(rmse)
print(r2)
print(rmset)
print(r2t)