
from config import data_info
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from config import *


def save_dict(data, name):
    with open('../' + name + '.txt', 'w') as f:  # dict转txt
        for k, v in data.items():
            f.write('\'' + k + '\'' + ':' + str(v) + ',')
            f.write('\n')
    f.close()


def data_read():
    # O
    if isinstance(data_info['year_data'], str):
        ...
    else:
        yearlist = data_info['year_data']
    # Mn
    fe_onehot_data = pd.read_excel(data_info['fea_data'], index_col=None)
    fsheet = fe_onehot_data.values
    fsheet = fsheet[1:, 1:].astype(np.float)
    fsheet[:, 2] = fsheet[:, 2] / 103.0
    fsheet[:, 3] = fsheet[:, 3] / 103.0
    fsheet[:, 4] = fsheet[:, 4] / 103.0
    return yearlist, fsheet


def extract_year(year, fsheet):
    mn_dict = {'time': str(year), 'data': []}
    f_rown = fsheet.shape[0]
    for j in range(0, f_rown):
        if float(fsheet[j, 0]) >= float(year):
            if float(fsheet[j, 1]) <= float(year):
                mn_dict['data'].append(j)
    return mn_dict


def read_data_all():
    o_dict = {}
    yearlist, fsheet = data_read()
    len_year = len(yearlist)

    for i in range(len_year):
        o_dict[str(i)] = extract_year(yearlist[i], fsheet)

    save_dict(o_dict, 'time_pz')
    return o_dict, yearlist, fsheet[:, :]
