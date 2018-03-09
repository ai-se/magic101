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

plt.figure(1)

plt.subplot(2,1,1)
plt.plot(backtolist("./data_file/miyazaki/abe0_miyazaki_mre.csv"))
plt.plot(backtolist("./data_file/miyazaki/random_miyazaki_mre.csv"))
plt.plot(backtolist("./data_file/miyazaki/de2_miyazaki_mre.csv"))
plt.plot(backtolist("./data_file/miyazaki/de8_miyazaki_mre.csv"))
# plt.plot(backtolist("./whigham_miyazaki_10.csv"))
plt.yscale('symlog')           # linear, log, symlog, logit
plt.ylabel('MRE')
# plt.xlim(110, 230)
plt.ylim(-1, )
plt.legend(['abe0', 'random', 'de2', 'de8'], loc='upper left')
# plt.xlabel('Dataset: miyazaki')
# plt.show()

plt.subplot(2,1,2)
plt.plot(backtolist("./data_file/miyazaki/abe0_miyazaki_sa.csv"))
plt.plot(backtolist("./data_file/miyazaki/random_miyazaki_sa.csv"))
plt.plot(backtolist("./data_file/miyazaki/de2_miyazaki_sa.csv"))
plt.plot(backtolist("./data_file/miyazaki/de8_miyazaki_sa.csv"))
# plt.plot(backtolist("./whigham_miyazaki_10.csv"))
plt.yscale('linear')           # linear, log, symlog, logit
plt.ylabel('SA')
# plt.xlim(110, 230)
plt.ylim(-1, 1.1)
plt.legend(['abe0', 'random', 'de2', 'de8'], loc='upper left')
plt.xlabel('Dataset: miyazaki')
plt.tight_layout()
plt.show()

