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

plt.subplot(3,2,1)
plt.plot(backtolist("./data_file/kemerer/abe0_kemerer_mre.csv"))
plt.plot(backtolist("./data_file/kemerer/random_kemerer_mre.csv"))
plt.plot(backtolist("./data_file/kemerer/de2_kemerer_mre.csv"))
plt.plot(backtolist("./data_file/kemerer/de8_kemerer_mre.csv"))
# plt.plot(backtolist("./whigham_albrecht_10.csv"))
plt.yscale('symlog')           # linear, log, symlog, logit
plt.ylabel('MRE')
# plt.xlim(110, 230)
plt.ylim(-1, )
plt.legend(['abe0', 'random', 'de2', 'de8'], loc='upper left')
plt.xlabel('Dataset: kemerer')
# plt.show()

plt.subplot(3,2,2)
plt.plot(backtolist("./data_file/kemerer/abe0_kemerer_sa.csv"))
plt.plot(backtolist("./data_file/kemerer/random_kemerer_sa.csv"))
plt.plot(backtolist("./data_file/kemerer/de2_kemerer_sa.csv"))
plt.plot(backtolist("./data_file/kemerer/de8_kemerer_sa.csv"))
# plt.plot(backtolist("./whigham_albrecht_10.csv"))
plt.yscale('symlog')           # linear, log, symlog, logit
plt.ylabel('SA')
# plt.xlim(110, 230)
plt.ylim(-1, 1)
plt.legend(['abe0', 'random', 'de2', 'de8'], loc='upper left')
plt.xlabel('Dataset: kemerer')
plt.tight_layout()
# plt.show()

plt.subplot(3,2,3)
plt.plot(backtolist("./data_file/maxwell/abe0_maxwell_mre.csv"))
plt.plot(backtolist("./data_file/maxwell/random_maxwell_mre.csv"))
plt.plot(backtolist("./data_file/maxwell/de2_maxwell_mre.csv"))
plt.plot(backtolist("./data_file/maxwell/de8_maxwell_mre.csv"))
# plt.plot(backtolist("./whigham_albrecht_10.csv"))
plt.yscale('symlog')           # linear, log, symlog, logit
plt.ylabel('MRE')
# plt.xlim(110, 230)
plt.ylim(-1, )
plt.legend(['abe0', 'random', 'de2', 'de8'], loc='upper left')
plt.xlabel('Dataset: maxwell')
# plt.show()

plt.subplot(3,2,4)
plt.plot(backtolist("./data_file/maxwell/abe0_maxwell_sa.csv"))
plt.plot(backtolist("./data_file/maxwell/random_maxwell_sa.csv"))
plt.plot(backtolist("./data_file/maxwell/de2_maxwell_sa.csv"))
plt.plot(backtolist("./data_file/maxwell/de8_maxwell_sa.csv"))
# plt.plot(backtolist("./whigham_albrecht_10.csv"))
plt.yscale('symlog')           # linear, log, symlog, logit
plt.ylabel('SA')
# plt.xlim(110, 230)
plt.ylim(-1, 1)
plt.legend(['abe0', 'random', 'de2', 'de8'], loc='upper left')
plt.xlabel('Dataset: maxwell')
# plt.show()

plt.subplot(3,2,5)
plt.plot(backtolist("./data_file/miyazaki/abe0_miyazaki_mre.csv"))
plt.plot(backtolist("./data_file/miyazaki/random_miyazaki_mre.csv"))
plt.plot(backtolist("./data_file/miyazaki/de2_miyazaki_mre.csv"))
plt.plot(backtolist("./data_file/miyazaki/de8_miyazaki_mre.csv"))
# plt.plot(backtolist("./whigham_albrecht_10.csv"))
plt.yscale('symlog')           # linear, log, symlog, logit
plt.ylabel('MRE')
# plt.xlim(110, 230)
plt.ylim(-1, )
plt.legend(['abe0', 'random', 'de2', 'de8'], loc='upper left')
plt.xlabel('Dataset: miyazaki')
# plt.show()

plt.subplot(3,2,6)
plt.plot(backtolist("./data_file/miyazaki/abe0_miyazaki_sa.csv"))
plt.plot(backtolist("./data_file/miyazaki/random_miyazaki_sa.csv"))
plt.plot(backtolist("./data_file/miyazaki/de2_miyazaki_sa.csv"))
plt.plot(backtolist("./data_file/miyazaki/de8_miyazaki_sa.csv"))
# plt.plot(backtolist("./whigham_albrecht_10.csv"))
plt.yscale('symlog')           # linear, log, symlog, logit
plt.ylabel('SA')
# plt.xlim(110, 230)
plt.ylim(-1, 1)
plt.legend(['abe0', 'random', 'de2', 'de8'], loc='upper left')
plt.xlabel('Dataset: miyazaki')
plt.tight_layout()
plt.show()