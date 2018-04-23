from __future__ import division,print_function
from sklearn.tree import DecisionTreeClassifier , DecisionTreeRegressor
import numpy as np

__author__ = 'panzer'


def cart_format(dataset, test, train):
    train_arr = train.as_matrix()
    test_arr = test.as_matrix()
    train_x = np.array([ x[:-1] for x in train_arr])
    train_y = np.array([ x[-1] for x in train_arr])
    test_x = [ x[:-1] for x in test_arr]
    test_y = [ x[-1] for x in test_arr]
    return train_x, train_y, test_x, test_y

def data_format(dataset, test, train):
    train_arr = train.as_matrix()
    test_arr = test.as_matrix()
    train_x = np.array([ x[:-1] for x in train_arr])
    train_y = np.array([ x[-1] for x in train_arr])
    test_x = [ x[:-1] for x in test_arr]
    test_y = [ x[-1] for x in test_arr]
    return
  # def indep(x):
  #   rets=[]
  #   indeps = x.cells[:len(dataset.indep)]
  #   for i,val in enumerate(indeps):
  #     if i not in dataset.ignores:
  #       rets.append(val)
  #   return rets
  # dep   = lambda x: x.cells[len(dataset.indep)]
  # train_input_set, train_output_set = [], []
  # test_input_set, test_output_set = [], []
  # for row in train:
  #   train_input_set+=[indep(row)]
  #   train_output_set+=[dep(row)]
  # for row in test:
  #   test_input_set += [indep(row)]
  #   test_output_set += [dep(row)]
  # return train_input_set, train_output_set, test_input_set, test_output_set

def cart(dataset, test, train):
    train_ip, train_op, test_ip, test_op = cart_format(dataset, test, train)
    dec_tree = DecisionTreeRegressor(criterion="mse", random_state=1)
    dec_tree.fit(train_ip,train_op)
    y_predict = dec_tree.predict(test_ip)
    return [y_predict, test_op, np.append(train_op,test_op)]