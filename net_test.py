import os
import shutil
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from data_creat import read_data_all
from data_read import datacreate
from config import test_opt
from net_model import Node_creat

import pandas as pd
import os

class Node_model():
    def __init__(self, savedir):
        super(Node_model, self).__init__()
        self.net = Node_creat()
        self.savedir = savedir

        self.y_index = [-1, -3,-5, -7, -14]
        self.yy = [self.tryloss(float(ii)) for ii in self.y_index]
        self.yyn=[0,-2,-4,-5,-7,-13]
        self.y_index2 = ['$\mathregular{10^{0}}$', '$\mathregular{10^{-2}}$', '$\mathregular{10^{-4}}$',
                         '$\mathregular{10^{-5}}$','$\mathregular{10^{-7}}$', '$\mathregular{10^{-13}}$']
        self.yy2 = [self.tryloss(np.log10(np.power(10,float(ii))*0.21005)) for ii in self.yyn]

        self.xx = [i*500 for i in range(9)]
        self.x_index = [str(i*0.5) for i in range(9)]
    # produce_function---------------------------------------------
    def tryloss(self, x):
        return -np.log(0-x)

    def lossback(self, y):
        return 0-(np.exp(-y))

    def fenkai(self, bestlab, logflag=''):
        labmin = []
        labmax = []
        bestlab = np.array(bestlab)
        sizee = bestlab.shape[0]
        for i in range(sizee):
            athed = bestlab[i, :]
            if logflag == 'logonly':
                labmin.append(np.log10(athed[0, 0]))
                labmax.append(np.log10(athed[0, 1]))
            elif logflag == 'log':
                labmin.append(self.tryloss(np.log10(athed[0, 0])))
                labmax.append(self.tryloss(np.log10(athed[0, 1])))
            elif logflag == 'exponly':
                labmin.append(np.exp(athed[0, 0]))
                labmax.append(np.exp(athed[0, 1]))
            elif logflag == 'exp':
                labmin.append(np.power(10,self.lossback(athed[0, 0])))
                labmax.append(np.power(10,self.lossback(athed[0, 1])))
            elif logflag == 'exlog10only':
                labmin.append(np.log10(np.exp(athed[0, 0])))
                labmax.append(np.log10(np.exp(athed[0, 1])))
            elif logflag == 'back':
                labmin.append(self.lossback(athed[0, 0]))
                labmax.append(self.lossback(athed[0, 1]))
            else:
                labmin.append(athed[0, 0])
                labmax.append(athed[0, 1])

        return labmin, labmax

    def load(self, loaddir):
        if self.device != 'cpu':
            self.net.cuda()
        #读入参数
        pr_dict = torch.load(loaddir, map_location=self.device)
        model_dict = self.net.state_dict()
        pretrained_dict = {k: v for k, v in pr_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.net.load_state_dict(model_dict)

    def repredict(self, draw=True):
        print('+====+====+====+====+====+====+====+====+====+====+====+====+')
        print('load data')
        o_dict, yearlist, featlist= read_data_all()
        test_loader = datacreate(o_dict, yearlist, featlist)
        val_bar = tqdm(test_loader)
        yearlist = []
        outputlist = []
        self.net.eval()

        print('+====+====+====+====+====+====+====+====+====+====+====+====+')
        print('test begin')
        with torch.no_grad():
            for step, data in enumerate(val_bar):
                feature, year = data
                ochangew = feature.to(self.device)
                outputs= self.net(ochangew)
                pml = torch.mean(outputs, dim=0).unsqueeze(dim=0)
                ochange = (pml).detach().cpu()
                yearlist.append(year.item())
                outputlist.append(np.array(ochange))
        outmin, outmax = self.fenkai(outputlist)

        print('+====+====+====+====+====+====+====+====+====+====+====+====+')
        print('save result')
        logmin, logmax = self.fenkai(outputlist, 'back')
        rawmin, rawmax = self.fenkai(outputlist, 'exp')
        sheet = []
        sheet.append(yearlist)
        sheet.append(logmin)
        sheet.append(logmax)
        sheet.append(rawmin)
        sheet.append(rawmax)
        df = pd.DataFrame(sheet)
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
        df.to_excel(self.savedir + '/out.xlsx', index=False)

        print('+====+====+====+====+====+====+====+====+====+====+====+====+')
        print('draw pic')
        if draw == True:
            fig = plt.figure(figsize=(6,3.63), dpi=300)
            ax = fig.add_subplot(1, 1, 1)
            ax2 = ax.twinx()
            plt.title('Predict result', fontsize=10)
            aa1 = ax.scatter(yearlist, outmin, color='tomato', label='Reconstruction $\mathregular{result_{min}}$')
            bb1 = ax.scatter(yearlist, outmax, color='royalblue', label='Reconstruction $\mathregular{result_{max}}$')
            ax.set_ylabel("Lg[$\it{p}$$\mathregular{O_{2}}$ (atm)] ")
            ax2.set_ylabel("$\it{p}$$\mathregular{ O_{2}}$ (PAL) ", rotation=-90, labelpad=12)
            plt.xticks([])
            ax.set_xticklabels([])
            ax.set_yticks(self.yy)
            ax.set_yticklabels(self.y_index)
            ax2.set_yticks(self.yy2)
            ax2.set_yticklabels(self.y_index2)
            ax.set_xlim((4000, 1))
            ax.set_ylim((-3, 1))
            ax2.set_ylim((-3, 1))
            plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0, wspace=0)

            if not os.path.exists(self.savedir):
                os.makedirs(self.savedir)
            plt.savefig(self.savedir + '/pic.tif', transparent=True,dpi=800)

    def test(self):
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
        # model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device != 'cpu':
            self.net.cuda()
        self.load(test_opt['model'] + test_opt['savename'])
        print('-------------------------------------------------')
        print(' predict begin')
        self.repredict()


def test_mn():
    savedir = test_opt['savedir']
    Net = Node_model(savedir)
    Net.test()

