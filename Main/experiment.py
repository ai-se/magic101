#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2018, Jianfeng Chen <jchen37@ncsu.edu>
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
from multiprocessing import Process

from data.new_data import data_albrecht, data_desharnais, data_finnish, data_kemerer, data_maxwell, data_miyazaki
from Main.methods import de_estimate, random_strategy, abe0_strategy
from utils.kfold import KFoldSplit_df


def DE2(TrainSet, TestSet):
    return de_estimate(2, TrainSet, TestSet)


def DE8(TrainSet, TestSet):
    return de_estimate(8, TrainSet, TestSet)


def DE28(TrainSet, TestSet):
    return de_estimate([2, 8], TrainSet, TestSet)


def RANDOM10(TrainSet, TestSet):
    return random_strategy(10, TrainSet, TestSet)


def RANDOM20(TrainSet, TestSet):
    return random_strategy(20, TrainSet, TestSet)


def ABE0(TrainSet, TestSet):
    return abe0_strategy(TrainSet, TestSet)


def WHIGHAM():
    pass


def exec(modelIndex, methodologyId):
    """
    :param modelIndex:
    :param methodologyId:
    :return: writing to final_list.txt
        ^^^^ repeatID mre sa
    """
    datafunc = [data_albrecht, data_desharnais, data_finnish, data_kemerer, data_maxwell, data_miyazaki]
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
        elif methodologyId == 1:
            res = RANDOM10(train, test)
        elif methodologyId == 2:
            res = RANDOM20(train, test)
        elif methodologyId == 3:
            res = DE2(train, test)
        elif methodologyId == 4:
            res = DE8(train, test)
        elif methodologyId == 5:
            res = DE28(train, test)
        time.sleep(random.random() * 2)  # avoid writing conflicts

        if methodologyId != 5:
            with open('final_list.txt', 'a+') as f:
                print("Finishing " + str(sys.argv))
                f.write(
                    str(modelIndex) + ';' + str(methodologyId) + ';' + str(res[0]) + ';' + str(res[1]) + ';' +
                    str(list(map(int, res[2].tolist()))) + '\n')
        else:  # running DE2/8
            with open('final_list.txt', 'a+') as f:
                print("Finishing " + str(sys.argv))
                f.write(
                    str(modelIndex) + ';' + '3' + ';' + str(res[0][0]) + ';' + str(res[0][1]) + ';' +
                    str(list(map(int, res[0][2].tolist()))) + '\n')
                f.write(
                    str(modelIndex) + ';' + '4' + ';' + str(res[1][0]) + ';' + str(res[1][1]) + ';' +
                    str(list(map(int, res[1][2].tolist()))) + '\n'
                )


def run():
    """
    system arguments:
        1 modelIndex,
        2 methodology ID [0-ABE0, 1-RANDOM10, 2-RANDOM20, 3-DE2, 4-DE8, 5-DE2/8]
        3 core Num, or the repeat times
    :return:
    """

    if len(sys.argv) > 1:
        modelIndex, methodologyId, repeatNum = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    else:  # for default local run
        modelIndex, methodologyId, repeatNum = 0, 3, 1
    time1 = time.time()
    p = list()
    for i in range(repeatNum):
        p.append(Process(target=exec, args=(modelIndex, methodologyId)))
        p[-1].start()

    for i in range(repeatNum):
        p[i].join()
    print("total time = " + str(time.time() - time1))


if __name__ == '__main__':
    # exec(0, 3)
    run()
