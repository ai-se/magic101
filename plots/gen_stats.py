import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import *

data_list = ["albrecht", "desharnais", "finnish", "kemerer", "maxwell",
             "miyazaki", "china", "isbsg10", "kitchenham"]

data_index = 3

def gen_stat(data_index):

    data = pd.read_csv('Outputs/final_list.txt', sep=";", header=None)
    data.columns = ["Data_ID", "Method_ID", "MRE", "SA", "CONFIG", "NGEN"]


    df_DE30 = data.query('Data_ID == ["'+str(data_index)+'"] and Method_ID == ["5"]')
    DE30_NGEN = sorted(df_DE30.loc[:,"NGEN"])


    df_GA100 = data.query('Data_ID == ["'+str(data_index)+'"] and Method_ID == ["6"]')
    GA100_NGEN = sorted(df_GA100.loc[:,"NGEN"])

    # print("Dataset:", data_list[data_index])
    # print("DE250_gen:", int(np.median(DE250_NGEN)))
    # print("GA250_gen:", int(np.median(GA250_NGEN)))

    GEN_DE = int(np.median(DE30_NGEN))
    GEN_GA = int(np.median(GA100_NGEN))
    return GEN_DE, GEN_GA

List_DE = []
List_GA = []

for i in range(9):
    List_DE.append(gen_stat(i)[0])
    List_GA.append(gen_stat(i)[1])

print(List_DE)
print(List_GA)


data = [List_DE, List_GA]
objects = ("albrecht", "desharnais", "finnish", "kemerer", "maxwell",
             "miyazaki", "china", "isbsg10", "kitchenham")
X = np.arange(9)
plt.xticks(X, objects)
plt.bar(X + 0.00, data[0], color = 'b', width = 0.25)
plt.bar(X + 0.25, data[1], color = 'r', width = 0.25)
plt.legend(['DE30', 'GA100'], loc='upper left', fontsize = 'small')
plt.title('Number of Generations for DE30 and GA100')

plt.show()