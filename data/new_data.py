import pandas as pd
from scipy.io.arff import loadarff


def data_albrecht():
    raw_data = loadarff("./data/albrecht.arff")
    df_data = pd.DataFrame(raw_data[0])
    return df_data


def data_china():
    raw_data = loadarff("./data/china.arff")
    df_data = pd.DataFrame(raw_data[0])
    new_chin = df_data.drop(columns=['ID', 'N_effort'])
    return new_chin


def data_desharnais():
    raw_data = loadarff("./data/desharnais.arff")
    df_data = pd.DataFrame(raw_data[0])
    new_desh = df_data.drop(index=[37, 43, 65, 74], columns=['Project', 'YearEnd', 'Envergure', 'PointsNonAjust', 'Language'])
    columnsTitles = ['TeamExp', 'ManagerExp', 'Length', 'Transactions', 'Entities', 'PointsAdjust', 'Effort']
    new_desh = new_desh.reindex(columns=columnsTitles)
    return new_desh


def data_finnish():
    raw_data = loadarff("./data/finnish.arff")
    df_data = pd.DataFrame(raw_data[0])
    new_finn = df_data.drop(columns=['ID'])
    columnsTitles = ['hw', 'at', 'FP', 'co', 'prod', 'lnsize', 'lneff', 'dev.eff.hrs.']
    new_finn = new_finn.reindex(columns=columnsTitles)
    return new_finn


def data_kemerer():
    raw_data = loadarff("./data/kemerer.arff")
    df_data = pd.DataFrame(raw_data[0])
    new_keme = df_data.drop(columns=['ID'])
    return new_keme


def data_maxwell():
    raw_data = loadarff("./data/maxwell.arff")
    df_data = pd.DataFrame(raw_data[0])
    new_maxw = df_data.drop(columns=['Syear'])
    return new_maxw


def data_miyazaki():
    raw_data = loadarff("./data/miyazaki94.arff")
    df_data = pd.DataFrame(raw_data[0])
    new_miya = df_data.drop(columns=['ID'])
    return new_miya

