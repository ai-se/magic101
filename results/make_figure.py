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
plt.subplot(321)
plt.plot(backtolist("./results/csv_file/rd_albrecht.csv"))
plt.plot(backtolist("./results/csv_file/de1_albrecht.csv"))
plt.plot(backtolist("./results/csv_file/de2_albrecht.csv"))
# plt.plot(backtolist("./results/csv_file/de4_albrecht.csv"))
plt.yscale('symlog')           # linear, log, symlog, logit
plt.ylabel('MRE')
plt.xlim(200, 475)
plt.ylim(-0.1, 20)
plt.legend(['random', 'de(gen=1)', 'de(gen=2)', 'de(gen=4)'], loc='upper left')
plt.xlabel('Dataset: albrecht')
# plt.show()

plt.subplot(323)
plt.plot(backtolist("./results/csv_file/rd_kemerer.csv"))
plt.plot(backtolist("./results/csv_file/de1_kemerer.csv"))
plt.plot(backtolist("./results/csv_file/de2_kemerer.csv"))
# plt.plot(backtolist("./results/csv_file/de4_kemerer.csv"))
plt.yscale('symlog')           # linear, log, symlog, logit
plt.ylabel('MRE')
plt.xlim(100, 300)
plt.ylim(-0.1, 20)
plt.legend(['random', 'de(gen=1)', 'de(gen=2)', 'de(gen=4)'], loc='upper left')
plt.xlabel('Dataset: kemerer')
# plt.show()

plt.subplot(325)
plt.plot(backtolist("./results/csv_file/rd_maxwell.csv"))
plt.plot(backtolist("./results/csv_file/de1_maxwell.csv"))
plt.plot(backtolist("./results/csv_file/de2_maxwell.csv"))
# plt.plot(backtolist("./results/csv_file/de4_maxwell.csv"))
plt.yscale('symlog')           # linear, log, symlog, logit
plt.ylabel('MRE')
plt.xlim(400, 1200)
plt.ylim(-0.1, 20)
plt.legend(['random', 'de(gen=1)', 'de(gen=2)', 'de(gen=4)'], loc='upper left')
plt.xlabel('Dataset: maxwell')
plt.tight_layout()
# plt.show()

plt.subplot(322)
plt.plot(backtolist("./results/csv_file/rd_albrecht.csv"))
plt.plot(backtolist("./results/csv_file/de1_albrecht.csv"))
plt.plot(backtolist("./results/csv_file/de2_albrecht.csv"))
plt.plot(backtolist("./results/csv_file/de4_albrecht.csv"))
plt.yscale('symlog')           # linear, log, symlog, logit
plt.ylabel('MRE')
plt.xlim(200, 475)
plt.ylim(-0.1, 20)
plt.legend(['random', 'de(gen=1)', 'de(gen=2)', 'de(gen=4)'], loc='upper left')
plt.xlabel('Dataset: albrecht')
# plt.show()

plt.subplot(324)
plt.plot(backtolist("./results/csv_file/rd_kemerer.csv"))
plt.plot(backtolist("./results/csv_file/de1_kemerer.csv"))
plt.plot(backtolist("./results/csv_file/de2_kemerer.csv"))
plt.plot(backtolist("./results/csv_file/de4_kemerer.csv"))
plt.yscale('symlog')           # linear, log, symlog, logit
plt.ylabel('MRE')
plt.xlim(100, 300)
plt.ylim(-0.1, 20)
plt.legend(['random', 'de(gen=1)', 'de(gen=2)', 'de(gen=4)'], loc='upper left')
plt.xlabel('Dataset: kemerer')
# plt.show()

plt.subplot(326)
plt.plot(backtolist("./results/csv_file/rd_maxwell.csv"))
plt.plot(backtolist("./results/csv_file/de1_maxwell.csv"))
plt.plot(backtolist("./results/csv_file/de2_maxwell.csv"))
plt.plot(backtolist("./results/csv_file/de4_maxwell.csv"))
plt.yscale('symlog')           # linear, log, symlog, logit
plt.ylabel('MRE')
plt.xlim(400, 1200)
plt.ylim(-0.1, 20)
plt.legend(['random', 'de(gen=1)', 'de(gen=2)', 'de(gen=4)'], loc='upper left')
plt.xlabel('Dataset: maxwell')
plt.tight_layout()
plt.show()