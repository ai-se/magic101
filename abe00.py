from __future__ import division
import numpy as np
import random
from scipy.io import arff
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler


def abe0(meta, trainData, testData, neigbors):
    """
    :param meta: attribute name
    :param trainData:
    :param testData:
    :param neighbors: parameters in KNN
    :return: precision
    """
    k = neigbors

    X = trainData[meta.names()[1:-1]]
    Y = trainData[meta.names()[-2:-1]]
    X = map(lambda x: list(x), X)
    Y = map(lambda x: list(x), Y)

    scaler1 = MinMaxScaler(copy=True, feature_range=(0, 1))
    scaler1.fit(X)
    xx = scaler1.transform(X)
    scaler2 = MinMaxScaler(copy=True, feature_range=(0, 100))
    scaler2.fit(Y)
    yy = scaler2.transform(Y)
    yy = [int(i/10) + 1 for i in yy]

    yy = np.ravel(yy)
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(xx, yy)

    # Testing
    X_test = testData[meta.names()[1:-1]]
    X_test = map(lambda i: list(i), X_test)
    xx_test = scaler1.transform(X_test)

    y_predict = clf.predict(xx_test)

    Y_test = testData[meta.names()[-2:-1]]
    Y_test = map(lambda i: list(i), Y_test)
    y_actual = scaler2.transform(Y_test)
    y_actual = [int(i / 10) + 1 for i in y_actual]

    m = 0
    for predict, actual in zip(y_predict, y_actual):
        if predict == actual:
            m += 1
    accuracy = m / (len(y_actual))

    return accuracy


def fold_validation(arff_file, learner, *args):
    """

    :param arff_file:
    :param learner:
    :param args:
    :return:
    """
    data, meta = arff.loadarff(arff_file)
    random.shuffle(data)
    indices = range(len(data))
    folds = 10 if len(data) > 50 else 3
    kf = KFold(n_splits=folds)

    accuracy_distribution = list()
    for train, test in kf.split(indices):
        trainData = data[train]
        testData = data[test]
        accuracy = learner(meta, trainData, testData, *args)
        accuracy_distribution.append(accuracy)

    return accuracy_distribution


if __name__ == '__main__':
    accuracy_dist  = fold_validation('china.arff', abe0, 5)
    print([round(i,2) for i in accuracy_dist])