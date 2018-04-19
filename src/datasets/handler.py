import os
import re
import sys
import arff
import pandas as pd
from glob2 import glob

root = os.path.join(os.getcwd().split("src")[0], "src")
if root not in sys.path:
    sys.path.append(root)


class Handler:
    def __init__(self): pass

    @classmethod
    def get_data(cls):
        all_ = {}
        files = glob(os.path.join(root, "datasets/*.arff"))
        for file_ in files:
            fname = re.sub(".arff", "", os.path.basename(file_))
            try: arff_file = arff.load(open(file_, "rb"))
            except: pass
            columns = [col[0] for col in arff_file["attributes"]]
            dframe = pd.DataFrame(arff_file['data'], columns=columns)
            all_.update({fname: dframe})

        return all_


if __name__ == "__main__":
    h = Handler.get_data()
