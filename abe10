from scipy import spatial

import numpy as np
from scipy.spatial import distance_matrix
from scipy.io import arff
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from Mantel import test

def file_process(arff_file):
    data, meta = arff.loadarff(arff_file)
    return data, meta

file_name = 'kemerer.arff'

A = file_process(file_name)
B = A[0]

B_0 = A[0][A[1].names()[1:-1]]
B_1 = A[0][A[1].names()[-1:]]
B_0 = map(lambda x: list(x), B_0)
B_1 = map(lambda x: list(x), B_1)
# print (B_1)
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
scaler.fit(B_0)
scaler.fit(B_1)
B_0 = scaler.transform(B_0)
B_1 = scaler.transform(B_1)
C_0 = pairwise_distances(B_0)
C_1 = pairwise_distances(B_1)
# # D_0 = np.matrix(C_0)
# D_1 = np.matrix(C_1)
#
# # boo = spatial.distance.is_valid_dm(D_0)
# boo = spatial.distance.is_valid_dm(C_0, tol=0.001)
# print (B_0)

ZZ = test(C_0, C_1, perms=1000, method='pearson', tail='lower')

print (ZZ)
