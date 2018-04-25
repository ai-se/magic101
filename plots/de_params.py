import pandas as pd
import os
import glob
import pdb
import numpy as np
import collections


def read_files(path="../"):
    def get_counts(data_name, df):
        res = {}
        tunelst = {
            "max_features": [0.01, 1],
            "max_depth": [1, 12],
            "min_samples_split": [2, 20],
            "min_samples_leaf": [1, 12],
        }
        f = lambda val, n: np.percentile(range(val[0], val[1]+1), n)
        out = data_name + "\n"
        # get actual depth

        for key, val in tunelst.items():
            # print(key)
            if key == "max_features":
                q1, q2, q3, q4 = [0.25, 0.5, 0.75, 1]
            else:
                q1 = f(val, 25)
                q2 = f(val, 50)
                q3 = f(val, 75)
                q4 = f(val, 100)
            bins = [q1, q2, q3, q4]
            if key in ["max_depth", "min_samples_leaf"]:
                bins = [3, 6, 9, 12]
                if key == "max_depth":
                    appear_counts = np.digitize(df["actual_depth"].values, bins, True)
                else:
                    appear_counts = np.digitize(df["min_samples_leaf"].values, bins, True)
            else:
                appear_counts = np.digitize(df[key].values, bins, True)
            temp = [0, 0, 0, 0]
            for a in appear_counts:
                if a >= 3:
                    a = 3
                temp[a] += 1
            # idx, counts = np.unique(x, return_counts=True)
            res[key] = map(lambda x: int(round(x,2)*100), np.array(temp)/sum(temp))
            # print(temp)
            # line = "&\dbox{{{0}}}\dbox{{{1}}}\dbox{{{2}}}\dbox{{{3}}}\n".format(*temp)
            line ="&"
            for val in res[key]:
                line += "\dbox{{{0:02d}}}".format(val) if val < 50 else "\wbox{{{0:02d}}}".format(val)
            line +="\n"
            # line = "&\dbox{{{0:02d}}}\dbox{{{1:02d}}}\dbox{{{2:02d}}}\dbox{{{3:02d}}}\n".format(*res[key])
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
