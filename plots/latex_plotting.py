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

import os

from plots.parse_outputs import reading

model_names = ['albrecht', 'desharnais', 'finnish', 'kemerer', 'maxwell', 'miyazaki', 'china', 'isbsg10', 'kitchenham']


def plot_latex_mre(model_index, model_name):
    reading(model_index, model_name)
    cmd = 'cat ' + model_name \
          + '_mre.txt| python2 utils/stats.py --text 30 --latex True | grep \'} \\\\\\\\$\' >> r.txt'
    with open('tmp.sh', 'w') as f:
        f.write(cmd)
    os.system(cmd)

    with open('r.txt', 'r') as myfile:
        data = myfile.read()
    os.system('rm *_*.txt r.txt tmp.sh')
    return data


def plot_latex_sa(model_index, model_name):
    reading(model_index, model_name)
    cmd = 'cat ' + model_name + '_sa.txt| python2 utils/stats.py --text 30 --latex True --higher True | grep \'} \\\\\\\\$\' >> r.txt'
    with open('tmp.sh', 'w') as f:
        f.write(cmd)
    os.system(cmd)

    with open('r.txt', 'r') as myfile:
        data = myfile.read()

    os.system('rm *_*.txt r.txt tmp.sh')
    return data


def plot_mre_for_all():
    """
    PRINTING TO A FILE mre-latex.txt
    :return:
    """
    f = open('Outputs/mre-latex.txt', 'w')
    for i, name in enumerate(model_names):
        print(name)
        P = plot_latex_mre(i, name)
        f.write('\\nm{' + name + '}\\\\' + '\n')
        f.write(P)
    f.close()


def plot_sa_for_all():
    """
        PRINTING TO A FILE sa-latex.txt
        :return:
        """
    f = open('Outputs/sa-latex.txt', 'w')
    for i, name in enumerate(model_names):
        print(name)
        P = plot_latex_sa(i, name)
        f.write('\\nm{' + name + '}\\\\' + '\n')
        f.write(P)
    f.close()


if __name__ == '__main__':
    plot_mre_for_all()
    plot_sa_for_all()
