from __future__ import division, print_function
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np
from Optimizer.tuner import *
from Optimizer.errors import mre


def cart_format(dataset, test, train):
    train_arr = train.as_matrix()
    test_arr = test.as_matrix()
    train_x = np.array([x[:-1] for x in train_arr])
    train_y = np.array([x[-1] for x in train_arr])
    test_x = [x[:-1] for x in test_arr]
    test_y = [x[-1] for x in test_arr]
    return train_x, train_y, test_x, test_y


def data_format(dataset):
    data_arr = dataset.as_matrix()
    data_x = np.array([x[:-1] for x in data_arr])
    data_y = np.array([x[-1] for x in data_arr])
    return data_x, data_y


def cart(test, train):
    train_ip, train_op = data_format(train)
    test_ip, test_op = data_format(test)
    dec_tree = DecisionTreeRegressor(criterion="mse", random_state=1)
    dec_tree.fit(train_ip, train_op)
    y_predict = dec_tree.predict(test_ip)
    return [y_predict, test_op]


def tune_learner(learner,
                 train_X,
                 train_Y,
                 tune_X,
                 tune_Y,
                 goal,
                 num_population,
                 repeats,
                 life,
                 target_class=None):
    """
  :param learner:
  :param train_X:
  :param train_Y:
  :param tune_X:
  :param tune_Y:
  :param goal:
  :param target_class:
  :return:
  """
    if not target_class:
        target_class = goal
    clf = learner(train_X, train_Y, tune_X, tune_Y, goal)
    tuner = DE_Tune_ML(clf, clf.get_param(), goal, target_class,
                       num_population=num_population,repeats=repeats, life=life)
    return tuner.Tune()


class Learners(object):
    def __init__(self, clf, train_X, train_Y, predict_X, predict_Y, goal,
                 random_seed):
        self.X_train = train_X
        self.y_train = train_Y
        self.X_tune = predict_X
        self.y_actual = predict_Y
        self.goal = goal
        self.param_distribution = self.get_param()
        self.learner = clf
        self.params = None
        self.scores = None
        self.random_seed = random_seed

    def learn(self, **kwargs):
        if "random_state" not in kwargs:
            kwargs["random_state"] = self.random_seed  # set random state
        self.learner.set_params(**kwargs)
        clf = self.learner.fit(self.X_train, self.y_train)
        y_predict = clf.predict(self.X_tune)
        self.scores = mre(self.y_actual, y_predict)
        # print(self.scores)
        return self.scores



class Cart(Learners):
    name = "CART"

    def __init__(self,
                 train_x,
                 train_y,
                 predict_x,
                 predict_y,
                 goal,
                 random_seed=1):
        clf = DecisionTreeRegressor()
        self.tunelst = {
            "max_features": [0.01, 1],
            "max_depth": [1, 50],
            "min_samples_split": [2, 20],
            "min_samples_leaf": [1, 20],
            "random_state": [1, 1]
        }
        super(Cart, self).__init__(clf, train_x, train_y, predict_x, predict_y,
                                   goal, random_seed)

    def get_param(self):
        return self.tunelst
