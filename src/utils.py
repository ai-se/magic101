import numpy


class Misc:
    def __init__(self): pass

    @classmethod
    def train_test_split(cls, dframe, ratio=0.66):
        mask = numpy.random.rand(len(dframe)) < ratio
        train = dframe[mask].reset_index()
        test = dframe[~mask].reset_index()
        return train, test
