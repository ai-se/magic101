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

import sys

from data.new_data import data_albrecht, data_desharnais, data_finnish, data_kemerer, data_maxwell, data_miyazaki
from results.methods import de_estimate, random_strategy, abe0_strategy
from utils.kfold import KFoldSplit_df


def DE2(TrainSet, TestSet):
    return de_estimate(2, TrainSet, TestSet)


def DE8(TrainSet, TestSet):
    return de_estimate(8, TrainSet, TestSet)


def RANDOM(TrainSet, TestSet):
    return random_strategy(40, TrainSet, TestSet)


def ABE0(TrainSet, TestSet):
    return abe0_strategy(TrainSet, TestSet)


def WHIGHAM():
    pass


def hpc():
    """
    - system auguments:
        1 modelIndex,
        2 methodology ID [0-DE2, 1-DE8, 2-RANDOM, 3-ABE0]
    :return: writing to sysout
        ^^^^ repeatID mre sa
    """
    print("RUNING with " + str(sys.argv))

    datafunc = [data_albrecht, data_desharnais, data_finnish, data_kemerer, data_maxwell, data_miyazaki]
    model = datafunc[int(sys.argv[1])]
    methodologyId = int(sys.argv[2])
    res = None
    for _ in range(2):
        for train, test in KFoldSplit_df(model(), 3):
            if methodologyId == 0:
                res = DE2(train, test)
            elif methodologyId == 1:
                res = DE8(train, test)
            elif methodologyId == 2:
                res = RANDOM(train, test)
            elif methodologyId == 3:
                res = ABE0(train, test)

            with open('FINAL.txt', 'a+') as f:
                f.write(
                    '^^^ ' + sys.argv[1] + ' ' + sys.argv[2] + ' ' + str(res[0]) + ' ' + str(res[1]) + '\n')


if __name__ == '__main__':
    hpc()
