from __future__ import division
from stats import Stat
import numpy as np
import random
from scipy.io import arff
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors.regression import KNeighborsRegressor, check_array, _get_weights
import pdb

class MedianKNNRegressor(KNeighborsRegressor):
    def predict(self, X):
        X = check_array(X, accept_sparse='csr')

        neigh_dist, neigh_ind = self.kneighbors(X)

        weights = _get_weights(neigh_dist, self.weights)

        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        ######## Begin modification
        if weights is None:
            y_pred = np.median(_y[neigh_ind], axis=1)
        else:
            # y_pred = weighted_median(_y[neigh_ind], weights, axis=1)
            raise NotImplementedError("weighted median")
        ######### End modification

        if self._y.ndim == 1:
            y_pred = y_pred.ravel()

        return y_pred

def abe5(meta, trainData, testData, neigbors, adm):
    """
    :param meta: attribute name
    :param trainData:
    :param testData:
    :param neighbors: parameters in KNN
    :return: precision
    """
    k = neigbors

    X = trainData[meta.names()[1:-1]]
    Y = trainData[meta.names()[-1:]]
    X = map(lambda x: list(x), X)
    Y = map(lambda x: list(x), Y)

    scaler1 = MinMaxScaler(copy=True, feature_range=(0, 1))
    scaler1.fit(X)
    xx = scaler1.transform(X)
    # scaler2 = MinMaxScaler(copy=True, feature_range=(0, 100))
    # scaler2.fit(Y)
    # yy = scaler2.transform(Y)
    # yy = [int(i/10) + 1 for i in yy]

    # yy = np.ravel(yy)
    # clf = KNeighborsClassifier(n_neighbors=k)
    # clf.fit(xx, yy)

    yy = np.ravel(Y)

    if adm == 1:
        clf = KNeighborsRegressor(n_neighbors=k, weights='uniform')
    elif adm == 2:
        clf = KNeighborsRegressor(n_neighbors=k, weights='distance')
    elif adm == 3:
        clf = MedianKNNRegressor(n_neighbors=k)
    else:
        print ("adm error")

    clf.fit(xx, yy)

    # Testing
    X_test = testData[meta.names()[1:-1]]
    X_test = map(lambda i: list(i), X_test)
    xx_test = scaler1.transform(X_test)

    y_predict = clf.predict(xx_test)

    Y_test = testData[meta.names()[-1:]]
    Y_test = map(lambda i: list(i), Y_test)

    y_actual = np.ravel(Y_test)

    MRE = abs(y_predict - y_actual)/y_actual
    MRE = np.around(MRE,decimals=3)
    # pdb.set_trace()
    # print([round(i, 2) for i in RE])
    return MRE.tolist()


    # y_actual = scaler2.transform(Y_test)
    # y_actual = [int(i / 10) + 1 for i in y_actual]
    #
    # m = 0
    # for predict, actual in zip(y_predict, y_actual):
    #     if predict == actual:
    #         m += 1
    # accuracy = m / (len(y_actual))
    #
    # return accuracy


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

    MRE_distribution = list()
    for train, test in kf.split(indices):
        trainData = data[train]
        testData = data[test]
        relative_error = learner(meta, trainData, testData, args[0], args[1])
        MRE_distribution.append(relative_error)

    MRE_distribution = [y for x in MRE_distribution for y in x]
    MRE_distribution.insert(0, 'ABE5(adm='+ str(args[1]) + ')')

    return MRE_distribution


if __name__ == '__main__':
    outputs = list()
    file_name = 'maxwell.arff'
    # method = 1
    # MRE_dist = fold_validation(file_name, abe5, 3, method)
    # outputs.append(MRE_dist)
    for method in range(1,4):
        MRE_dist = fold_validation(file_name, abe5, 3, method)
        outputs.append(MRE_dist)
        print(MRE_dist)

    # for i in outputs:
    #     print (i)

    print (file_name)
    # print (outputs)
    Stat.rdivDemo(outputs)