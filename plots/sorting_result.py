import pandas as pd
import pdb
import numpy as np
import matplotlib.pyplot as plt
from random import *

# [0-albrecht, 1-desharnais, 2-finnish, 3-kemerer, 4-maxwell, 5-miyazaki, 6-china, 7-isbsg10, 8-kitchenham]
from sympy.physics.quantum.circuitplot import matplotlib

data_index = 7

data_list = ["albrecht", "desharnais", "finnish", "kemerer", "maxwell",
             "miyazaki", "china", "isbsg10", "kitchenham"]

# data = pd.read_csv('./final_list_cr0.3_f0.8.txt', sep=";", header=None)
data = pd.read_csv('Outputs/final_list_ABE0.txt', sep=";", header=None)

data.columns = ["Data_ID", "Method_ID", "MRE", "SA", "CONFIG", "NGEN"]

whigham = pd.read_csv('Outputs/ATLM.txt', sep=";", header=None)
whigham.columns = ["Data", "Method", "MRE", "SA"]

df25 = whigham.query('Data == ["'+str(data_list[data_index])+'"] and Method == ["ATLM"]')
df25_MRE = sorted(df25.loc[:,"MRE"])
df25_SA = sorted(df25.loc[:,"SA"])
# df15_MRE = [x/4+0.5 for x in df15_MRE]
# df15_SA = [x/3+0.2 for x in df15_SA]


# df10 = data.query('Data_ID == ["'+str(data_index)+'"] and Method_ID == ["0"]')
# df10_MRE = sorted(df10.loc[:,"MRE"])
# df10_SA = sorted(df10.loc[:,"SA"])
#
#
# df11 = data.query('Data_ID == ["'+str(data_index)+'"] and Method_ID == ["1"]')
# df11_MRE = sorted(df11.loc[:,"MRE"])
# df11_SA = sorted(df11.loc[:,"SA"])
#
#
# df12 = data.query('Data_ID == ["'+str(data_index)+'"] and Method_ID == ["2"]')
# df12_MRE = sorted(df12.loc[:,"MRE"])
# df12_SA = sorted(df12.loc[:,"SA"])
#
#
# df13 = data.query('Data_ID == ["'+str(data_index)+'"] and Method_ID == ["3"]')
# df13_MRE = sorted(df13.loc[:,"MRE"])
# df13_SA = sorted(df13.loc[:,"SA"])
#
#
# df14 = data.query('Data_ID == ["'+str(data_index)+'"] and Method_ID == ["4"]')
# df14_MRE = sorted(df14.loc[:,"MRE"])
# df14_SA = sorted(df14.loc[:,"SA"])


df15 = data.query('Data_ID == ["'+str(data_index)+'"] and Method_ID == ["5"]')
df15_MRE = sorted(df15.loc[:,"MRE"])
df15_SA = sorted(df15.loc[:,"SA"])


df16 = data.query('Data_ID == ["'+str(data_index)+'"] and Method_ID == ["6"]')
df16_MRE = sorted(df16.loc[:,"MRE"])
df16_SA = sorted(df16.loc[:,"SA"])

# print(len(data.query('Data_ID == ["6"] and Method_ID == ["6"]')))
# print(data.query('Data_ID == ["6"] and Method_ID == ["8"]'))
# print(len(data.query('Data_ID == ["6"]')))
aaa = (data.query('Method_ID == ["0"]').iloc[:,3])
print(type(aaa))
# font = {
#         # 'weight' : 'bold',
#         'size'   : 14}
#
# matplotlib.rc('font', **font)
#
#
# plt.figure(1)
# plt.subplot(1, 2, 1)
# plt.title('Smaller is better', fontsize=22)
# # plt.plot(df10_MRE)
# # plt.plot(df11_MRE)
# # plt.plot(df12_MRE)
# # plt.plot(df13_MRE)
# # plt.plot(df14_MRE)
# # plt.plot(df25_MRE)
# plt.plot(df15_MRE)
# plt.plot(df16_MRE)
# plt.yscale('linear')           # linear, log, symlog, logit
# plt.ylabel('MRE', fontsize=22)
# plt.ylim(-0.1, 1.25)
# # plt.yticks([0,0.25,0.5,0.75,1])
# plt.legend(['DE250', 'GA250'], loc='lower right', fontsize = 'small')
# # plt.xlabel('Dataset: desharnais', fontsize=30)
# # plt.show()
#
# plt.subplot(1, 2, 2)
# plt.title('Larger is better', fontsize=22)
# # plt.plot(df10_SA)
# # plt.plot(df11_SA)
# # plt.plot(df12_SA)
# # plt.plot(df13_SA)
# # plt.plot(df14_SA)
# # plt.plot(df25_SA)
# plt.plot(df15_SA)
# plt.plot(df16_SA)
# plt.yscale('linear')           # linear, log, symlog, logit
# plt.ylabel('SA', fontsize=22)
# plt.ylim(-0.1, 1.25)
# # plt.yticks([0,0.25,0.5,0.75,1])
# plt.legend(['DE250', 'GA250'], loc='upper left', fontsize = 'small')
# # plt.xlabel('Dataset: desharnais', fontsize=30)
# plt.show()