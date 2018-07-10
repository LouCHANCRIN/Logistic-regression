import pandas as pd
import numpy as np
import math as ma
import cmath as cma

data = pd.read_csv("ex2data1.csv")
line, col = np.shape(data)
Y = data["admission"]
Y = np.reshape(Y, (line, 1))
X = [np.insert(row, 0, 1) for row in data.drop(["admission"], axis=1).values]
SX = [np.insert(row, 0, 1) for row in data.drop(["admission"], axis=1).values]
X = np.reshape(X, (line, col))
SX = np.reshape(SX, (line, col))
theta = [[0.0] * col]
theta = np.reshape(theta, (col, 1))

q = 0
_min = [[0.0] * col]
_min = np.reshape(_min, (col, 1))
_max = [[0.0] * col]
_max = np.reshape(_max, (col, 1))
_mean = [[0.0] * col]
_mean = np.reshape(_mean, (col, 1))

def scale(X):
    for j in range(1, col):
        _min[j] = X[0][j]
        _max[j] = X[0][j]
    for i in range(0, line):
        for j in range(1, col):
            if (X[i][j] < _min[j]):
                _min[j] = X[i][j]
            if (X[i][j] > _max[j]):
                _max[j] = X[i][j]
            _mean[j] += X[i][j]
    for j in range(1, col):
        _mean[j] /= line
    for i in range(0, line):
        for j in range(1, col):
            X[i][j] = (X[i][j] - _mean[j]) / (_max[j] - _min[j])
    return (X)

X = scale(X)

def hypothese(Xk, theta):
    ret = 0
    TT = np.transpose(-theta)
    e = Xk.dot(theta)
    ret = 1 / (1 + (np.exp(-e)))
    TT = np.transpose(-theta)
    #print(ret)
    return (ret)

def cost(X, Y, theta, j):
    a = 0
    for k in range(0, line):
        hyp = hypothese(X[k], theta)
        res =  (1 / 2) *(-Y[k] * np.log10(hyp)) - ((1 - Y[k]) * np.log10(1 - hyp))
        res = (res - Y[k]) * X[k][j]
        a += res
    return (a)

def logistic_reg(X, Y, theta, alpha, num_iters):
    temp = [[0.0] * col]
    temp = np.reshape(temp, (col, 1))
    for i in range(0, num_iters):
        if (i % 100 == 0):
            print(i)
        for j in range(0, col):
            temp[j] = theta[j] - ((alpha / line) * cost(X, Y, theta, j))
        for j in range(0, col):
            theta[j] = temp[j]
    return (theta)

def scale_theta(theta, _min, _max, _mean):
    for j in range(1, col):
        theta[j] = (theta[j] - _mean[j]) / (_max[j] - _min[j])
    return (theta)

alpha = 0.01
num_iters = 2500
theta = logistic_reg(X, Y, theta, alpha, num_iters)
print(theta)
theta = scale_theta(theta, _min, _max, _mean) * -1
print(theta)

def precision(SX, Y, line, col, theta):
    a = 0.0
    for i in range(0, line):
        result = theta[0] + theta[1] * SX[i][1] + theta[2] * SX[i][2]
        if ((result / 2 >= 50 and Y[i] == 1) or (result / 2 < 50 and Y[i] == 0)):
            a += 1
        print(result)
    return (a)

a = precision(SX, Y, line, col, theta)
print(a)
print(line)
print((a / line) * 100)
