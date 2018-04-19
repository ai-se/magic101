
import os
import pandas as pd
import arff
import numpy as np
from glob2 import glob

class Handler:
    def get_data(self):
        all = {}
        files = glob(os.path.realpath())