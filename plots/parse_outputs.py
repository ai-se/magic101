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


import pandas as pd

method_names = ['ABE0', 'RANDOM40', 'RANDOM160', 'DE2', 'DE8']


def reading(model_index, model_name):
    """
    0, albrecht
    1, desharnais
    ...
    :param model_index:
    :param model_name:
    :return: No results. Writing two files for latex_plotting.py
    """
    if type(model_index) is int:
        model_index = str(model_index)

    data = pd.read_csv('./final_list_cr0.3_f0.8.txt', sep=";", header=None)
    data.columns = ["Data_ID", "Method_ID", "MRE", "SA", "CONFIG"]

    whigham = pd.read_csv('./ATLM.txt', sep=";", header=None)
    whigham.columns = ["Data", "Method", "MRE", "SA"]
    df15 = whigham.query('Data == ["' + model_name + '"] and Method == ["ATLM"]')

    df15_MRE = ' '.join([str(i) for i in sorted(df15.loc[:, "MRE"])])
    df15_SA = ' '.join([str(i) for i in sorted(df15.loc[:, "SA"])])

    mre_file_name = model_name + '_mre.txt'
    sa_file_name = model_name + '_sa.txt'

    with open(mre_file_name, 'a+') as f:
        f.write('ATLM' + '\n' + df15_MRE)

    with open(sa_file_name, 'a+') as f:
        f.write('ATLM' + '\n' + df15_SA)

    for methodid in range(5):
        df = data.query('Data_ID == ["' + model_index + '"] and Method_ID == ["' + str(methodid) + '"]')
        mre_str = ' '.join([str(i) for i in sorted(df.loc[:, "MRE"])])
        sa_str = ' '.join([str(i) for i in sorted(df.loc[:, "SA"])])

        with open(mre_file_name, 'a+') as f:
            f.write('\n\n' + method_names[methodid] + '\n' + mre_str)

        with open(sa_file_name, 'a+') as f:
            f.write('\n\n' + method_names[methodid] + '\n' + sa_str)


if __name__ == '__main__':
    reading('0', 'albrecht')
