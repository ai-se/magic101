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


import ast

import numpy as np
import pandas as pd


def most_comm(lst):
    return max(((item, lst.count(item)) for item in set(lst)), key=lambda a: a[1])[0]


def most_common(lst):
    return max(lst, key=lst.count)


def counting(data_id):
    data = pd.read_csv('Outputs/final_list_cr0.3_f0.8.txt', sep=";", header=None)
    data.columns = ["Data_ID", "Method_ID", "MRE", "SA", "CONFIG"]
    new_data = data.query('Data_ID == ["' + str(data_id) +
                          '"] and Method_ID == ["' + str(4) +
                          '"]')
    new_data = new_data.loc[:, "CONFIG"]
    alist = []
    for i in new_data:
        i = ast.literal_eval(i)
        alist.append(i)

    alist = np.array(alist).T

    density = list()

    # handing subsets
    d = alist[0].tolist()
    n = len(d)
    density.append([(d.count(0) + d.count(2)) / n, d.count(1) / n])

    # handing weighing
    d = alist[1].tolist()
    n = len(d)
    density.append([d.count(i) / n for i in range(8)])

    # handing disct
    d = alist[2].tolist()
    n = len(d)
    density.append([d.count(0) / n,
                    d.count(1) / n,
                    (d.count(2) + d.count(3) + d.count(4)) / n,
                    ])

    # handing similarity
    d = alist[3].tolist()
    n = len(d)
    density.append([d.count(i) / n for i in range(6)])

    # handing adaption
    d = alist[4].tolist()
    n = len(d)
    density.append([d.count(i) / n for i in range(4)])

    # handing analogies
    d = alist[5].tolist()
    n = len(d)
    density.append([d.count(i) / n for i in range(6)])

    return density


def latex_plot(model_name, data_id):
    def print_one_box(den):
        den *= 100
        if den == 100:
            den = 99
        if den >= 50:
            return "\wbox{" + str(int(den)) + "}"
        elif den < 10:
            return "\dbox{0" + str(int(den)) + "}"
        else:
            return "\dbox{" + str(int(den)) + "}"

    density = counting(data_id)
    str1 = model_name + '&'
    for i, v in enumerate(density):
        # handing v here
        v = [float(i) / sum(v) for i in v]
        for j, vv in enumerate(v):
            str1 += print_one_box(vv)
        str1 += '&'
    str1 = str1[:-1]
    str1 += "\\\\"
    print(str1)


def latex_plot_all_density():
    model_names = ['albrecht', 'desharnais', 'finnish', 'kemerer', 'maxwell', 'miyazaki', 'china', 'isbsg10',
                   'kitchenham']
    for i, n in enumerate(model_names):
        latex_plot(n, i)


if __name__ == '__main__':
    latex_plot_all_density()
