import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import *


data = pd.read_csv('./final_list.txt', sep=";", header=None)
data.columns = ["Data_ID", "Method_ID", "MRE", "SA", "CONFIG"]

whigham = pd.read_csv('./ATLM.txt', sep=";", header=None)
whigham.columns = ["Data", "Method", "MRE", "SA"]
# print(whigham)

df15 = whigham.query('Data == ["isbsg10"] and Method == ["ATLM"]')
df15_MRE = sorted(df15.loc[:,"MRE"])
df15_SA = sorted(df15.loc[:,"SA"])

# print(len(data.query('Data_ID == ["7"]')))

# dataset = albrecht, method = DE2

df10 = data.query('Data_ID == ["7"] and Method_ID == ["0"]')
df10_MRE = sorted(df10.loc[:,"MRE"])
df10_SA = sorted(df10.loc[:,"SA"])
#
# np.savetxt("./desharnais_MRE_ABE0.txt", df10_MRE, newline=" ", fmt='%s')
#
df11 = data.query('Data_ID == ["7"] and Method_ID == ["1"]')
df11_MRE = sorted(df11.loc[:,"MRE"])
df11_SA = sorted(df11.loc[:,"SA"])
#
# np.savetxt("./desharnais_MRE_RANDOM40.txt", df11_MRE, newline=" ", fmt='%s')
#
df12 = data.query('Data_ID == ["7"] and Method_ID == ["2"]')
df12_MRE = sorted(df12.loc[:,"MRE"])
df12_SA = sorted(df12.loc[:,"SA"])
#
# np.savetxt("./desharnais_MRE_RANDOM160.txt", df12_MRE, newline=" ", fmt='%s')
#
df13 = data.query('Data_ID == ["7"] and Method_ID == ["3"]')
df13_MRE = sorted(df13.loc[:,"MRE"])
df13_SA = sorted(df13.loc[:,"SA"])
#
# np.savetxt("./desharnais_MRE_DE2.txt", df13_MRE, newline=" ", fmt='%s')
#
df14 = data.query('Data_ID == ["7"] and Method_ID == ["4"]')
df14_MRE = sorted(df14.loc[:,"MRE"])
df14_SA = sorted(df14.loc[:,"SA"])
#
# np.savetxt("./desharnais_MRE_DE8.txt", df14_MRE, newline=" ", fmt='%s')
#
#

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
# plt.yticks([0,0.25,0.5,0.75,1])                                            I
plt.legend(['ABE0', 'RD40', 'RD160', 'DE2', 'DE8', 'ATLM'], loc='upper left', fontsize = 'small')
# plt.xlabel('Dataset: desharnais', fontsize=30)
# plt.xlabel('Dataset: miyazaki')
plt.show()
