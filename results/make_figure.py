import csv
import matplotlib.pyplot as plt


def backtolist(file):
    reader = open(file, "rt")
    read = csv.reader(reader)
    list_1 = list()
    for row in read:
        list_1 += row

    list_2 = list()
    for item in list_1:
        list_2.append(float(item))

    list_3 = sorted(list_2)
    return list_3


plt.plot(backtolist("rd_kemerer.csv"))
plt.plot(backtolist("de1_kemerer.csv"))
plt.yscale('symlog')           # linear, log, symlog, logit
plt.ylabel('MRE')
plt.xlim(120, 300)
plt.ylim(0.01, 20)
plt.legend(['rd', 'de1'], loc='upper left')
# plt.xlabel('Dataset: maxwell')
plt.show()