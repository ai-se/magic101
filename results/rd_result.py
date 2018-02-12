
from __future__ import division

import logging
import sys
import pdb

import pandas as pd
import numpy as np
from scipy.io import arff

from Optimizer.bridge import ft_dict_to_ABE_setting
from ABE.main import abe_execute
from FeatureModel.Feature_tree import FeatureTree
from utils.kfold import KFoldSplit


def random_config(ft, dataset):
    """
    Randomly generate an ABE hyperparameter.
    To reduce the error, use cross-validation (10 fold).

    :param ft:
    :param dataset:
    :return: average relative error [0,1]
    """

    while True:
        X = ft.top_down_random(None)

        if ft.check_fulfill_valid(X):
            break
        logging.debug('=== Invalid configuration. Regenerating...')
    settings = ft_dict_to_ABE_setting(X)
    # print(settings)

    data0, meta0 = arff.loadarff(dataset)
    all_error = list()
    for meta, train, test in KFoldSplit(dataset, folds=len(data0)):
        trainData = pd.DataFrame(data=train)
        testData = pd.DataFrame(data=test)
        error = abe_execute(S=settings, train=trainData, test=testData)
        all_error.append(error)
    return all_error


if __name__ == '__main__':
    # logging.basicConfig(stream=sys.stdout,
    #                     format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    #                     level=logging.DEBUG)

    url = "./FeatureModel/tree_model.xml"
    ft = FeatureTree()
    ft.load_ft_from_url(url)

    repeats = 20
    gen_list = list()
    for _ in range(repeats):
        gen_list += random_config(ft, "data/maxwell.arff")
    print(len(gen_list))
    print(gen_list)

    np.savetxt("rd_maxwell2.csv", gen_list, delimiter=",", fmt='%s')

