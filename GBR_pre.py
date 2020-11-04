#!/usr/bin/env python
# -*- coding:utf-8 -*-
#author: xhwan

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


data_train = np.loadtxt('data_train.csv', delimiter=",", dtype="float")
pre = np.loadtxt('pre.csv', delimiter=",", dtype="float")
ss = MinMaxScaler()
x = data_train[..., 0:20]
x = ss.fit_transform(x)
p = pre[..., 0:20]
p = ss.fit_transform(p)
y = data_train[..., 20]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=16)

params = {'n_estimators': 500,
          'max_depth': 5,
          'min_samples_split': 5,
          'learning_rate': 0.005,
          'loss': 'huber'}

model = GBR(**params)
model.fit(x_train,y_train)

rmse = np.sqrt(mse(y_train, model.predict(x_train)))
r2 = r2_score(y_train, model.predict(x_train))

prediction = model.predict(p)
print('pre:', model.predict(x_test))
np.savetxt('prediction.csv', prediction, delimiter=",")
print(rmse)
print(r2)
