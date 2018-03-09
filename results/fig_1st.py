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
plt.plot(backtolist("./data_file/albrecht/abe0_albrecht_mre.csv"))
plt.plot(backtolist("./data_file/albrecht/random_albrecht_mre.csv"))
plt.plot(backtolist("./data_file/albrecht/de2_albrecht_mre.csv"))
plt.plot(backtolist("./data_file/albrecht/de8_albrecht_mre.csv"))
# plt.plot(backtolist("./whigham_albrecht_10.csv"))
plt.yscale('symlog')           # linear, log, symlog, logit
plt.ylabel('MRE')
# plt.xlim(110, 230)
plt.ylim(-1, )
plt.legend(['abe0', 'random', 'de2', 'de8'], loc='upper left')
plt.xlabel('Dataset: albrecht')
# plt.show()

plt.subplot(3,2,2)
plt.plot(backtolist("./data_file/albrecht/abe0_albrecht_sa.csv"))
plt.plot(backtolist("./data_file/albrecht/random_albrecht_sa.csv"))
plt.plot(backtolist("./data_file/albrecht/de2_albrecht_sa.csv"))
plt.plot(backtolist("./data_file/albrecht/de8_albrecht_sa.csv"))
# plt.plot(backtolist("./whigham_albrecht_10.csv"))
plt.yscale('symlog')           # linear, log, symlog, logit
plt.ylabel('SA')
# plt.xlim(110, 230)
plt.ylim(-1, 1)
plt.legend(['abe0', 'random', 'de2', 'de8'], loc='upper left')
plt.xlabel('Dataset: albrecht')
plt.tight_layout()
# plt.show()

plt.subplot(3,2,3)
plt.plot(backtolist("./data_file/desharnais/abe0_desharnais_mre.csv"))
plt.plot(backtolist("./data_file/desharnais/random_desharnais_mre.csv"))
plt.plot(backtolist("./data_file/desharnais/de2_desharnais_mre.csv"))
plt.plot(backtolist("./data_file/desharnais/de8_desharnais_mre.csv"))
# plt.plot(backtolist("./whigham_albrecht_10.csv"))
plt.yscale('symlog')           # linear, log, symlog, logit
plt.ylabel('MRE')
# plt.xlim(110, 230)
plt.ylim(-1, )
plt.legend(['abe0', 'random', 'de2', 'de8'], loc='upper left')
plt.xlabel('Dataset: desharnais')
# plt.show()

plt.subplot(3,2,4)
plt.plot(backtolist("./data_file/desharnais/abe0_desharnais_sa.csv"))
plt.plot(backtolist("./data_file/desharnais/random_desharnais_sa.csv"))
plt.plot(backtolist("./data_file/desharnais/de2_desharnais_sa.csv"))
plt.plot(backtolist("./data_file/desharnais/de8_desharnais_sa.csv"))
# plt.plot(backtolist("./whigham_albrecht_10.csv"))
plt.yscale('symlog')           # linear, log, symlog, logit
plt.ylabel('SA')
# plt.xlim(110, 230)
plt.ylim(-1, 1)
plt.legend(['abe0', 'random', 'de2', 'de8'], loc='upper left')
plt.xlabel('Dataset: desharnais')
# plt.show()

plt.subplot(3,2,5)
plt.plot(backtolist("./data_file/finnish/abe0_finnish_mre.csv"))
plt.plot(backtolist("./data_file/finnish/random_finnish_mre.csv"))
plt.plot(backtolist("./data_file/finnish/de2_finnish_mre.csv"))
plt.plot(backtolist("./data_file/finnish/de8_finnish_mre.csv"))
# plt.plot(backtolist("./whigham_albrecht_10.csv"))
plt.yscale('symlog')           # linear, log, symlog, logit
plt.ylabel('MRE')
# plt.xlim(110, 230)
plt.ylim(-1, )
plt.legend(['abe0', 'random', 'de2', 'de8'], loc='upper left')
plt.xlabel('Dataset: finnish')
# plt.show()

plt.subplot(3,2,6)
plt.plot(backtolist("./data_file/finnish/abe0_finnish_sa.csv"))
plt.plot(backtolist("./data_file/finnish/random_finnish_sa.csv"))
plt.plot(backtolist("./data_file/finnish/de2_finnish_sa.csv"))
plt.plot(backtolist("./data_file/finnish/de8_finnish_sa.csv"))
# plt.plot(backtolist("./whigham_albrecht_10.csv"))
plt.yscale('symlog')           # linear, log, symlog, logit
plt.ylabel('SA')
# plt.xlim(110, 230)
plt.ylim(-1, 1)
plt.legend(['abe0', 'random', 'de2', 'de8'], loc='upper left')
plt.xlabel('Dataset: finnish')
plt.tight_layout()
plt.show()

