import pandas as pd
import numpy as np

data = pd.read_csv("ex2data1.csv")
line, col = np.shape(data)

def scale(X, _min, _max, _mean):
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

def scale_theta(theta, _min, _max, _mean):
    for j in range(1, col):
        theta[j] = (theta[j] - _mean[j]) / (_max[j] - _min[j])
    return (theta)

def get_data(data, _min, _max, _mean):
    Y = data["admission"]
    Y = np.reshape(Y, (line, 1))
    X = [np.insert(row, 0, 1) for row in data.drop(["admission"], axis=1).values]
    unscaled_X = [np.insert(row, 0, 1) for row in data.drop(["admission"], axis=1).values]
    X = np.reshape(X, (line, col))
    unscaled_X = np.reshape(unscaled_X, (line, col))
    X = scale(X, _min, _max, _mean)
    return (X, Y, unscaled_X)

def precision(unscaled_X, Y, theta):
    a = 0
    for i in range(0, line):
        result = theta[0] + theta[1] * unscaled_X[i][1] + theta[2] * unscaled_X[i][2]
        if ((result / 2 >= 50 and Y[i] == 1) or (result / 2 < 50 and Y[i] == 0)):
            a += 1
    return (a)

def hypothese(X, k, theta):
    TT = np.transpose(-theta)
    e = X[k].dot(theta)
    ret = 1 / (1 + (np.exp(-e)))
    return (ret)

def cost(X, Y, theta, j):
    a = 0
    for k in range(0, line):
        hyp = hypothese(X, k, theta)
        res =  (1 / 2) * (-Y[k] * np.log10(hyp)) - ((1 - Y[k]) * np.log10(1 - hyp))
        res = (res - Y[k]) * X[k][j]
        a += res
    return (a)

def logistic_reg(X, Y, theta, alpha):
    temp = np.reshape([[0.0] * col], (col, 1))
    for i in range(0, 2000):
        if (i % 100 == 0):
            print(i)
        for j in range(0, col):
            temp[j] = theta[j] - ((alpha / line) * cost(X, Y, theta, j))
        for j in range(0, col):
            theta[j] = temp[j]
    return (theta)

def main():
    alpha = 0.02
    _min = np.reshape([[0.0] * col], (col, 1))
    _max = np.reshape([[0.0] * col], (col, 1))
    _mean = np.reshape([[0.0] * col], (col, 1))
    X, Y, unscaled_X = get_data(data, _min, _max, _mean)

    theta = np.reshape([[0.0] * col], (col, 1))
    theta = logistic_reg(X, Y, theta, alpha)
    theta = scale_theta(theta, _min, _max, _mean) * -1

    a = precision(unscaled_X, Y, theta)
    print("precision:", a / line)

if __name__ == "__main__":
    main()
