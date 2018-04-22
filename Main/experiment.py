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
from multiprocessing import Process
from Main.methods import testing
from Main.methods import de_estimate, ga_estimate, random_strategy, nsga2_estimate
from data.new_data import data_albrecht, data_desharnais, data_finnish, data_kemerer, data_maxwell, data_miyazaki, \
    data_china, data_isbsg10, data_kitchenham
from utils.kfold import KFoldSplit_df
from Optimizer.feature_link import calc_error


def DE2(TrainSet, TestSet):
    best_config, ngen = de_estimate(20, 2, data=TrainSet)
    mre, sa, ci = calc_error(best_config, TestSet)
    return {"mre": mre, "sa": sa, "config": best_config, "gen": ngen}


def DE8(TrainSet, TestSet):
    best_config, ngen = de_estimate(20, 8, data=TrainSet)
    mre, sa, ci = calc_error(best_config, TestSet)
    return {"mre": mre, "sa": sa, "config": best_config, "gen": ngen}


def RANDOM10(TrainSet, TestSet):
    best_config = random_strategy(10, data=TrainSet)
    mre, sa, ci = calc_error(best_config, TestSet)
    return {"mre": mre, "sa": sa, "config": best_config}


def RANDOM30(TrainSet, TestSet):
    best_config = random_strategy(30, data=TrainSet)
    mre, sa, ci = calc_error(best_config, TestSet)
    return {"mre": mre, "sa": sa, "config": best_config}


def ABE0(TrainSet, TestSet):
    best_config = [0, 0, 0, 0, 0, 0]
    mre, sa, ci = calc_error(best_config, TestSet)
    return {"mre": mre, "sa": sa, "config": best_config}


def DE30(TrainSet, TestSet):
    best_config, ngen = de_estimate(30, 250, data=TrainSet)
    mre, sa, ci = calc_error(best_config, TestSet)
    return {"mre": mre, "sa": sa, "config": best_config, "gen": ngen}


def GA100(TrainSet, TestSet):
    best_config, ngen = ga_estimate(100, 250, data=TrainSet)
    mre, sa, ci = calc_error(best_config, TestSet)
    return {"mre": mre, "sa": sa, "config": best_config, "gen": ngen}


def DE10(TrainSet, TestSet):
    best_config, ngen = de_estimate(10, 250, data=TrainSet)
    mre, sa, ci = calc_error(best_config, TestSet)
    return {"mre": mre, "sa": sa, "config": best_config, "gen": ngen}


def NSGA2(TrainSet, TestSet):
    best_config, ngen = nsga2_estimate(100, 250, data=TrainSet)
    mre, sa, ci = calc_error(best_config, TestSet)
    return {"mre": mre, "sa": sa, "config": best_config, "gen": ngen}


def ATLM():
    pass


def exec(modelIndex, methodologyId):
    """
    :param modelIndex:
    :param methodologyId:
    :return: writing to final_list.txt
        ^^^^ repeatID mre sa
    """
    datafunc = [data_albrecht, data_desharnais, data_finnish, data_kemerer, data_maxwell, data_miyazaki,
                data_china, data_isbsg10, data_kitchenham]
    model = datafunc[modelIndex]
    res = None

    num_pj = len(model())
    if num_pj < 40:
        fold_num = 3
    else:
        fold_num = 10

    for train, test in KFoldSplit_df(model(), fold_num):
        if methodologyId == 0:
            res = ABE0(train, test)
        elif methodologyId == 9:
            res = RANDOM10(train, test)
        elif methodologyId == 10:
            res = RANDOM30(train, test)
        elif methodologyId == 11:
            res = DE2(train, test)
        elif methodologyId == 12:
            res = DE8(train, test)
        elif methodologyId == 5:
            res = DE30(train, test)
        elif methodologyId == 6:
            res = GA100(train, test)
        elif methodologyId == 7:
            res = DE10(train, test)
        elif methodologyId == 8:
            res = NSGA2(train, test)
        time.sleep(random.random() * 2)  # avoid writing conflicts

        if methodologyId == 0 or methodologyId == 9 or methodologyId == 10:
            with open('final_list.txt', 'a+') as f:
                # print("Finishing " + str(sys.argv))
                f.write(
                    str(modelIndex) + ';' + str(methodologyId) + ';' + str(res["mre"]) + ';' + str(res["sa"]) + ';' +
                    str(res["config"]) + ';' + '\n')
        else:
            with open('final_list.txt', 'a+') as f:
                f.write(
                    str(modelIndex) + ';' + str(methodologyId) + ';' + str(res["mre"]) + ';' + str(res["sa"]) + ';' +
                    str(res["config"]) + ';' + str(res["gen"]) + '\n')

def run():
    """
    system arguments:
        1 modelIndex [0-albrecht, 1-desharnais, 2-finnish, 3-kemerer, 4-maxwell, 5-miyazaki, 6-china, 7-isbsg10, 8-kitchenham]
        2 methodology ID [0-ABE0, 1-ATLM, 2-CART, 3-CoGEE, 4-MOEAD, 5-DE30, 6-GA100, 7-DE10, 8-NSGA2, 9-RD10, 10-RD30, 11-DE2, 12-DE8]
        3 core Num, or the repeat times
    :return:
    """
    start_time = time.time()

    if len(sys.argv) > 1:
        modelIndex, methodologyId, repeatNum = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    else:  # for default local run
        modelIndex, methodologyId, repeatNum = 2, 11, 1

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
