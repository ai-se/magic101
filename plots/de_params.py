import pandas as pd
import os
import glob
import pdb
import numpy as np
import collections


def read_files(path="../Results"):
    def get_counts(data_name, df):
        res = {}
        tunelst = {
            "max_features": [0.01, 1],
            "max_depth": [1, 50],
            "min_samples_split": [2, 20],
            "min_samples_leaf": [1, 20],
        }
        f = lambda val, n: np.percentile(range(val[0], val[1]+1), n)
        out = data_name + "\n"
        for key, val in tunelst.items():
            print(key)
            if key == "max_features":
                q1, q2, q3, q4 = [0.25, 0.5, 0.75, 1]
            else:
                q1 = f(val, 25)
                q2 = f(val, 50)
                q3 = f(val, 75)
                q4 = f(val, 100)
            bins = [q1, q2, q3, q4]
            appear_counts = np.digitize(df[key].values, bins, True)
            temp = [0, 0, 0, 0]
            for a in appear_counts:
                temp[a] += 1
            # idx, counts = np.unique(x, return_counts=True)
            res[key] = temp
            # print(temp)
            # line = "&\dbox{{{0}}}\dbox{{{1}}}\dbox{{{2}}}\dbox{{{3}}}\n".format(*temp)
            line = "&{0}&{1}&{2}&{3}".format(*temp)
            out += line
        print(out+"\\\\")
        # print("\n"+"="*20+"\n")

    csv_files = glob.glob(os.path.join(path, "*.csv"))
    print(csv_files)
    for one in csv_files:
        data_name = os.path.basename(one).split(".")[0][:-2]
        df = pd.read_csv(one)
        get_counts(data_name, df)


if __name__ == "__main__":
    read_files()
