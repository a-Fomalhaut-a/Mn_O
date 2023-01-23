
import torch
from torch import nn
from config import model_opt,data_info
from torch.autograd import Variable

class full_connect_res(nn.Module):
    def __init__(self):
        super(full_connect_res, self).__init__()
        self.netpara = model_opt['full_connect_res']
        data_info['fea_dim'] = data_info['fea_dim']
        self.classifier = nn.Sequential(
            nn.Linear(data_info['fea_dim'], self.netpara['out1']),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.Dropout(p=self.netpara['drop_ratio']),
            )
        self.last = nn.Sequential(
            nn.Linear(self.netpara['out2'], model_opt['outlast'])
            )
        self.kuai = []
        for i in range(self.netpara['layernum']):
            zz = nn.Sequential(
                nn.Linear(self.netpara['out1'], self.netpara['out2']),
                nn.LeakyReLU(negative_slope=0.01, inplace=False),
                nn.Dropout(p=self.netpara['drop_ratio']),
            )
            setattr(self, 'zz%i' % i, zz)
            self.kuai.append(zz)

    def forward(self, x):
        x = torch.squeeze(x)
        out1 = self.classifier(x)
        for i in range(self.netpara['layernum']):
            out2 = self.kuai[i](out1)
            out1 = out1+out2
        out = self.last(out1)
        return out