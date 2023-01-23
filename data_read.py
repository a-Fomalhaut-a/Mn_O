import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class ValDataset(Dataset):
    '''
    输出矿物特征，权重，一个时间
    '''
    def __init__(self, o_dict, yearlist, featlist):
        self. o_dict = o_dict
        self. yearlist = yearlist
        self.featlist = featlist
        self.size = len(self.featlist)

    def contfeat(self, nlist, fealist):
        newfea = []
        for i in nlist:
            newfea.append(fealist[int(i), :])
        return np.array(newfea)

    def featadd_year(self, nlist, fealist, yearla):
        fea = []
        for i in nlist:
            zz = (fealist[int(i), :]).tolist()
            zz.append((yearla-fealist[int(i), 1])/(yearla+0.01))
            zz.append((-yearla+fealist[int(i), 0])/(fealist[int(i), 0]+0.01))
            fea.append(zz)
        return np.array(fea)

    def one_max(self, featlist, canshu):
        featlist[:, 0] = np.log(featlist[:, 0]/canshu+1)
        featlist[:, 1] = np.log(featlist[:, 1] / canshu + 1)
        return featlist

    def __getitem__(self, index):
        xuhao = index
        year = self.yearlist[xuhao]
        assert (str(self.yearlist[xuhao]) == str(self.o_dict[str(xuhao)]['time'])), "对应{}有误".format(xuhao)
        year_xh = self.o_dict[str(xuhao)]['data']
        newfea = self.featadd_year(year_xh, self.featlist, year)
        newfea = self.one_max(newfea, 3000)
        feature = torch.from_numpy(newfea).float()

        assert (feature.isnan().sum() == 0), "特征{}有误".format(xuhao)
        return feature, year

    def __len__(self):
        return len(self.yearlist)


def datacreate(o_dict, yearlist, featlist):
    dataset = ValDataset(o_dict, yearlist, featlist)
    datatr = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)
    return datatr
