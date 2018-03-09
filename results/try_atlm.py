from __future__ import division, print_function
import sys, os, math

sys.dont_write_bytecode = True
sys.path.append(os.path.abspath("."))
import pandas
import statsmodels.formula.api as smf
from scipy.stats.stats import skew
import warnings

warnings.filterwarnings("ignore")
__author__ = 'panzer'


def make_formula(dataset):
    def get_levels(i):
        levels = map(str, set([row.cells[i] for row in dataset._rows]))
        return "[%s]" % (",".join(levels))

    labels = []
    for index, col_name in enumerate(dataset.indep):
        if dataset.is_continuous[index]:
            labels.append(col_name)
        else:
            labels.append("C(%s, levels=%s)" % (col_name, get_levels(index)))
    formula = "%s ~ %s" % (dataset.less[0], " + ".join(labels))
    return formula


def atlm(dataset, test_rows, rows):
    headers = dataset.indep + dataset.less
    train_data = [row.cells[:] for row in rows]
    continuous_variables = [decision for decision in dataset.decisions if dataset.is_continuous[decision]]
    transforms = get_transform_funcs(train_data, continuous_variables)
    for row in train_data:
        transform_row(row, continuous_variables, transforms)
    df = pandas.DataFrame(train_data, columns=headers)
    ols = smf.ols(formula=make_formula(dataset), data=df)
    lin_model = ols.fit()
    test_data = []
    for test_row in test_rows:
        test_data.append(transform_row(test_row.cells[:], continuous_variables, transforms))
    df_test = pandas.DataFrame(test_data, columns=headers)
    return lin_model.predict(df_test)


def get_transform_funcs(train, cols):
    transform_funcs = []
    for col in cols:
        vector = [row[col] for row in train]
        transforms = [
            (skew(vector, bias=False), "none"),
            (skew(log_transform(vector), bias=False), "log"),
            (skew(sqrt_transform(vector), bias=False), "sqrt")
        ]
        best_transform = sorted(transforms)[0][1]
        transform_funcs.append(best_transform)
    return transform_funcs


def transform_row(row, cols, transforms):
    for col, transform in zip(cols, transforms):
        if transform == "log":
            if row[col] <= 0:
                row[col] = -sys.maxint
            else:
                row[col] = math.log(row[col])
        elif transform == "sqrt":
            row[col] = math.sqrt(row[col])
        elif transform == "none":
            continue
        else:
            raise RuntimeError("Unknown transformation type : %s" % transform)
    return row


def log_transform(vector):
    transforms = []
    for one in vector:
        if one == 0:
            transforms.append(-float("inf"))
        else:
            transforms.append(math.log(one))
    return transforms


def sqrt_transform(vector):
    return [math.sqrt(one) for one in vector]


def make_columnar(rows, columns):
    column_data = []
    for column in columns:
        column_data.append([row[column] for row in rows])
    return column_data


def get_column(rows, column_index):
    return [row[column_index] for row in rows]

