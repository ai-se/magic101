import numpy as np
from numpy import genfromtxt

my_data = genfromtxt("./data_file/miyazaki/random_miyazaki_sa.csv", delimiter=' ', dtype=None)

np.savetxt("./data_file/random_miyazaki_sa.txt", my_data, newline=" ", fmt='%s')