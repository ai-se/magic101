#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2018, Jianfeng Chen <jchen37@ncsu.edu>, Tianpei Xia <txia4@ncsu.edu>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.
import random
import sys
import time
import pdb
import os
from multiprocessing import Process

import numpy

from Main.methods import de_estimate, ga_estimate, random_strategy, nsga2_estimate, moead_estimate
from Main.methods import testing
from Optimizer.feature_link import calc_error1
from data.new_data import data_albrecht, data_desharnais, data_finnish, data_kemerer, data_maxwell, data_miyazaki, \
    data_china, data_isbsg10, data_kitchenham
from utils.kfold import KFoldSplit_df
from sklearn.model_selection import train_test_split
from Main.cart import *
# from Optimizer.feature_link import  mre_calc, msa
from Optimizer.errors import  msa, mre
import numpy as np
import pandas as pd

f = lambda x: [ s[-1] for s in x.as_matrix()]

def DE2(AllSet, TrainSet, TestSet):
    best_config, ngen = de_estimate(20, 2, data=TrainSet)
    mre, sa, ci = calc_error1(best_config, TestSet, f(AllSet))
    return {"mre": mre, "sa": sa, "config": best_config, "gen": ngen}


def DE8(AllSet, TrainSet, TestSet):
    best_config, ngen = de_estimate(20, 8, data=TrainSet)
    mre, sa, ci = calc_error1(best_config, TestSet, f(AllSet))
    return {"mre": mre, "sa": sa, "config": best_config, "gen": ngen}


def RANDOM10(AllSet, TrainSet, TestSet):
    best_config = random_strategy(10, data=TrainSet)
    mre, sa, ci = calc_error1(best_config, TestSet, f(AllSet))
    return {"mre": mre, "sa": sa, "config": best_config}


def RANDOM30(AllSet, TrainSet, TestSet):
    best_config = random_strategy(30, data=TrainSet)
    mre, sa, ci = calc_error1(best_config, TestSet, f(AllSet))
    return {"mre": mre, "sa": sa, "config": best_config}


def ABE0(AllSet, TrainSet, TestSet):
    best_config = [0, 0, 0, 0, 0, 0]
    mre, sa, ci = calc_error1(best_config, TestSet, f(AllSet))
    return {"mre": mre, "sa": sa, "config": best_config}


def DE30(AllSet, TrainSet, TestSet):
    best_config, ngen = de_estimate(30, 250, data=TrainSet)
    mre, sa, ci = calc_error1(best_config, TestSet, f(AllSet))
    return {"mre": mre, "sa": sa, "config": best_config, "gen": ngen}


def GA100(AllSet, TrainSet, TestSet):
    best_config, ngen = ga_estimate(100, 250, data=TrainSet)
    mre, sa, ci = calc_error1(best_config, TestSet, f(AllSet))
    return {"mre": mre, "sa": sa, "config": best_config, "gen": ngen}


def DE10(AllSet, TrainSet, TestSet):
    best_config, ngen = de_estimate(10, 250, data=TrainSet)
    mre, sa, ci = calc_error1(best_config, TestSet, f(AllSet))
    return {"mre": mre, "sa": sa, "config": best_config, "gen": ngen}


def NSGA2(AllSet, TrainSet, TestSet):
    best_config, ngen = nsga2_estimate(NP=100, NGEN=250, data=TrainSet)
    mre, sa, ci = calc_error1(best_config, TestSet, f(AllSet))
    return {"mre": mre, "sa": sa, "config": best_config, "gen": ngen}


def MOEAD(AllSet, TrainSet, TestSet):
    best_config, ngen = moead_estimate(NP=4, NGEN=250, data=TrainSet)
    mre, sa, ci = calc_error1(best_config, TestSet, f(AllSet))
    return {"mre": mre, "sa": sa, "config": best_config, "gen": ngen}


def ATLM():
    pass


def CART0(dataset, Trainset, TestSet):
    Y_predict, Y_actual = cart(Trainset, TestSet)
    _mre, sa = mre(Y_predict, Y_actual, f(dataset)), msa(Y_predict, Y_actual, f(dataset))
    # print("mre: {0}, sa: {1}".format(_mre,sa))
    return {"mre": _mre, "sa": sa, "config": None, "gen": None}

def CART_DE2(dataset, Trainset, TestSet):
    learner = Cart
    goal = "MRE"
    train_X, train_Y= data_format(Trainset)
    new_train_X, tune_X, new_train_Y, tune_Y = train_test_split(train_X, train_Y, test_size=0.3, shuffle=True)
    test_X, Y_actual = data_format(TestSet)
    params, evaluation = tune_learner(learner, new_train_X, new_train_Y,
                                      tune_X, tune_Y, goal, num_population=20,repeats=2,life=20)
    clf = learner(train_X, train_Y, test_X, Y_actual, goal).learner.set_params(**params)
    clf.fit(train_X, train_Y)
    Y_predict = clf.predict(test_X)
    _mre, sa = mre(Y_predict, Y_actual, f(dataset)), msa(Y_predict, Y_actual, f(dataset))
    # print("mre: {0}, sa: {1}".format(_mre,sa))
    return {"mre": _mre, "sa": sa, "config": None, "gen": None, "params":params}

def CART_DE8(dataset, Trainset, TestSet):
    learner = Cart
    goal = "MRE"
    train_X, train_Y= data_format(Trainset)
    new_train_X, tune_X, new_train_Y, tune_Y = train_test_split(train_X, train_Y, test_size=0.3, shuffle=True)
    test_X, Y_actual = data_format(TestSet)
    params, evaluation = tune_learner(learner, new_train_X, new_train_Y, tune_X, tune_Y, goal, num_population=20, repeats=8,life=20)
    clf = learner(train_X, train_Y, test_X, Y_actual, goal).learner.set_params(**params)
    clf.fit(train_X, train_Y)
    Y_predict = clf.predict(test_X)
    _mre, sa = mre(Y_predict, Y_actual, f(dataset)), msa(Y_predict, Y_actual, f(dataset))
    # print("mre: {0}, sa: {1}".format(_mre,sa))
    params["actual_depth"] = clf.tree_.max_depth
    return {"mre": _mre, "sa": sa, "config": None, "gen": None, "params":params}

def CART_DE10(dataset, Trainset, TestSet):
    learner = Cart
    goal = "MRE"
    train_X, train_Y= data_format(Trainset)
    new_train_X, tune_X, new_train_Y, tune_Y = train_test_split(train_X, train_Y, test_size=0.3, shuffle=True)
    test_X, Y_actual = data_format(TestSet)
    params, evaluation = tune_learner(learner, new_train_X, new_train_Y, tune_X, tune_Y, goal, num_population=10,repeats=250,life=5)
    clf = learner(train_X, train_Y, test_X, Y_actual, goal).learner.set_params(**params)
    clf.fit(train_X, train_Y)
    Y_predict = clf.predict(test_X)
    _mre, sa = mre(Y_predict, Y_actual, f(dataset)), msa(Y_predict, Y_actual, f(dataset))
    # print("mre: {0}, sa: {1}".format(_mre,sa))
    return {"mre": _mre, "sa": sa, "config": None, "gen": None}

def CART_DE30(dataset, Trainset, TestSet):
    learner = Cart
    goal = "MRE"
    train_X, train_Y= data_format(Trainset)
    new_train_X, tune_X, new_train_Y, tune_Y = train_test_split(train_X, train_Y, test_size=0.3, shuffle=True)
    test_X, Y_actual = data_format(TestSet)
    params, evaluation = tune_learner(learner, new_train_X, new_train_Y, tune_X, tune_Y, goal,num_population=30,repeats=250,life=5)
    clf = learner(train_X, train_Y, test_X, Y_actual, goal).learner.set_params(**params)
    clf.fit(train_X, train_Y)
    Y_predict = clf.predict(test_X)
    _mre, sa = mre(Y_predict, Y_actual, f(dataset)), msa(Y_predict, Y_actual, f(dataset))
    # print("mre: {0}, sa: {1}".format(_mre,sa))
    return {"mre": _mre, "sa": sa, "config": None, "gen": None}

def get_best_param(stats, this_best):
    """

    :param stats: stats of one specific data set
    :param this_best: current best parameters for one DE tuning
    :return:
    """

    for key, val in this_best.items():
        stats[key] = stats.get(key,[]) + [val]
    return stats



def exec(modelIndex, methodologyId, save_params=True):
    """
    :param modelIndex:
    :param methodologyId:
    :return: writing to final_list.txt
        ^^^^ repeatID mre sa
    """
    numpy.random.seed()

    datafunc = [data_albrecht, data_desharnais, data_finnish, data_kemerer, data_maxwell, data_miyazaki,
                data_china, data_isbsg10, data_kitchenham]
    model = datafunc[modelIndex]
    res = None

    num_pj = len(model())
    if num_pj < 40:
        fold_num = 3
    else:
        fold_num = 10
    # temp = {}
    stats = {}
    data_name = datafunc[modelIndex].__name__.split("_")[1]
    for train, test in KFoldSplit_df(model(), fold_num):
        if methodologyId == 0:
            res = ABE0(model(), train, test)
        if methodologyId == 2:
            res = CART0(model(), train, test)
        elif methodologyId == 4:
            res = MOEAD(model(), train, test)
        elif methodologyId == 5:
            res = DE30(model(), train, test)
        elif methodologyId == 6:
            res = GA100(model(), train, test)
        elif methodologyId == 7:
            res = DE10(model(), train, test)
        elif methodologyId == 8:
            res = NSGA2(model(), train, test)
        elif methodologyId == 9:
            res = RANDOM10(model(), train, test)
        elif methodologyId == 10:
            res = RANDOM30(model(), train, test)
        elif methodologyId == 11:
            res = DE2(model(), train, test)
        elif methodologyId == 12:
            res = DE8(model(), train, test)
        elif methodologyId == 13:
            res = CART_DE2(model(), train, test)
        elif methodologyId == 14:
            res = CART_DE8(model(), train, test)
        elif methodologyId == 15:
            res = CART_DE10(model(), train, test)
        elif methodologyId == 16:
            res = CART_DE30(model(), train, test)
        time.sleep(random.random() * 2)  # avoid writing conflicts
        stats[data_name] = get_best_param(stats.get(data_name,{}), res["params"])

        if methodologyId == 0 or methodologyId == 9 or methodologyId == 10:
            with open('final_list.txt', 'a+') as f:
                # print("Finishing " + str(sys.argv))
                f.write(
                    str(modelIndex) + ';' + str(methodologyId) + ';' + str(res["mre"]) + ';' + str(res["sa"]) + ';' +
                    str(res["config"]) + ';' + '\n')
        elif methodologyId in [2, 13, 14, 15, 16]:
            with open('final_Cart_DE.txt','a+') as f:
                f.write(
                    str(modelIndex) + ';' + str(methodologyId) + ';' + str(res["mre"]) + ';' + str(res["sa"]) + ';' + '\n')

        else:
            with open('final_list.txt', 'a+') as f:
                f.write(
                    str(modelIndex) + ';' + str(methodologyId) + ';' + str(res["mre"]) + ';' + str(res["sa"]) + ';' +
                    str(res["config"]) + ';' + str(res["gen"]) + '\n')

    df = pd.DataFrame.from_dict(stats[data_name])
    if save_params:
        name = data_name+str(methodologyId)+".csv"
        if os.path.exists(name):
            with open(name, "a") as f:
                df.to_csv(f, header=False, index=False)
        else:
            with open(name, "a") as f:
                df.to_csv(f, header=True, index=False)


def run():
    """
    system arguments:
        1 modelIndex [0-albrecht, 1-desharnais, 2-finnish, 3-kemerer, 4-maxwell, 5-miyazaki, 6-china, 7-isbsg10, 8-kitchenham]
        2 methodology ID [0-ABE0, 1-ATLM, 2-CART0, 3-CoGEE, 4-MOEAD, 5-DE30, 6-GA100,
                          7-DE10, 8-NSGA2, 9-RD10, 10-RD30, 11-DE2, 12-DE8, 13-CART_DE2,
                          14-CART_DE8, 15-CART_DE10,16-CART_DE30]
        3 core Num, or the repeat times
    :return:
    """
    start_time = time.time()

    if len(sys.argv) > 1:
        modelIndex, methodologyId, repeatNum = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    else:  # for default local run
        modelIndex, methodologyId, repeatNum = 0, 0, 1

    if repeatNum == 1:
        time2 = time.time()
        exec(modelIndex, methodologyId)
        print("total time = " + str(time.time() - time2))
        sys.exit(0)

    time1 = time.time()
    p = list()
    for i in range(repeatNum):
        p.append(Process(target=exec, args=(modelIndex, methodologyId)))
        p[-1].start()

    for i in range(repeatNum):
        p[i].join()
    print("total time = " + str(time.time() - time1))
    print("--- %s seconds ---" % (time.time() - start_time))


def run_testing():
    """
    Debugging for a specific dataset under specified configuration indices.
    repeat 100 times for 3 folds
    :return:
    """
    for _ in range(100):
        for train, test in KFoldSplit_df(data_isbsg10(), 3):
            mre, sa, p = (testing(train, test, [1, 3, 2, 0, 0, 3]))
            print(mre)
            print(sa)


if __name__ == '__main__':
    run()
    # run_testing()
