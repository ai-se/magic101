import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import *

# [0-albrecht, 1-desharnais, 2-finnish, 3-kemerer, 4-maxwell, 5-miyazaki, 6-china, 7-isbsg10, 8-kitchenham]
# data = pd.read_csv('./final_list_cr0.3_f0.8.txt', sep=";", header=None)
data = pd.read_csv('Outputs/final_list_cr0.3_f0.8.txt', sep=";", header=None)
data.columns = ["Data_ID", "Method_ID", "MRE", "SA", "CONFIG"]

whigham = pd.read_csv('./ATLM.txt', sep=";", header=None)
whigham.columns = ["Data", "Method", "MRE", "SA"]
# print(whigham)

df15 = whigham.query('Data == ["kitchenham"] and Method == ["ATLM"]')
df15_MRE = sorted(df15.loc[:,"MRE"])
df15_SA = sorted(df15.loc[:,"SA"])

data_index = 8

df10 = data.query('Data_ID == ["'+str(data_index)+'"] and Method_ID == ["0"]')
df10_MRE = sorted(df10.loc[:,"MRE"])
df10_SA = sorted(df10.loc[:,"SA"])


df11 = data.query('Data_ID == ["'+str(data_index)+'"] and Method_ID == ["1"]')
df11_MRE = sorted(df11.loc[:,"MRE"])
df11_SA = sorted(df11.loc[:,"SA"])


df12 = data.query('Data_ID == ["'+str(data_index)+'"] and Method_ID == ["2"]')
df12_MRE = sorted(df12.loc[:,"MRE"])
df12_SA = sorted(df12.loc[:,"SA"])


df13 = data.query('Data_ID == ["'+str(data_index)+'"] and Method_ID == ["3"]')
df13_MRE = sorted(df13.loc[:,"MRE"])
df13_SA = sorted(df13.loc[:,"SA"])


df14 = data.query('Data_ID == ["'+str(data_index)+'"] and Method_ID == ["4"]')
df14_MRE = sorted(df14.loc[:,"MRE"])
df14_SA = sorted(df14.loc[:,"SA"])







plt.figure(1)

plt.plot(df10_MRE)
plt.plot(df11_MRE)
plt.plot(df12_MRE)
plt.plot(df13_MRE)
plt.plot(df14_MRE)
plt.plot(df15_MRE)
plt.yscale('linear')           # linear, log, symlog, logit
# plt.ylabel('MRE', fontsize=30, fontweight='bold')
plt.ylabel('MRE')
plt.ylim(-0.1, 1.25)
# plt.yticks([0,0.25,0.5,0.75,1])
plt.legend(['ABE0', 'RD40', 'RD160', 'DE2', 'DE8', 'ATLM'], loc='upper left', fontsize = 'small')
# plt.xlabel('Dataset: desharnais', fontsize=30)
# plt.xlabel('Dataset: miyazaki')
plt.show()

plt.figure(2)

plt.plot(df10_SA)
plt.plot(df11_SA)
plt.plot(df12_SA)
plt.plot(df13_SA)
plt.plot(df14_SA)
plt.plot(df15_SA)
plt.yscale('linear')           # linear, log, symlog, logit
# plt.ylabel('MRE', fontsize=30, fontweight='bold')
plt.ylabel('SA')
plt.ylim(-0.1, 1.25)
# plt.yticks([0,0.25,0.5,0.75,1])
plt.legend(['ABE0', 'RD40', 'RD160', 'DE2', 'DE8', 'ATLM'], loc='upper left', fontsize = 'small')
# plt.xlabel('Dataset: desharnais', fontsize=30)
# plt.xlabel('Dataset: miyazaki')
plt.show()