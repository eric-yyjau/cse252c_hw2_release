import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np


class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, m = 0.35):
        super(CustomLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features) )
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m

    def forward(self, input):
        x = input   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # size=B
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

        cos_theta = x.mm(ww) # size=(B,Classnum)
        cos_theta = cos_theta / torch.clamp(xlen.view(-1,1) * wlen.view(1,-1), min=1e-8 )
        cos_theta = cos_theta.clamp(-1,1)

        # IMPLEMENT phi_theta

        output = (cos_theta,phi_theta)
        return output


class CustomLoss(nn.Module):
    def __init__(self, s=64 ):
        super(CustomLoss, self).__init__()
        self.s = s

    def forward(self, input, target):
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(B,1)

        # IMPLEMENT loss

        _, predictedLabel = torch.max(cos_theta.data, 1)
        predictedLabel = predictedLabel.view(-1, 1)
        accuracy = (predictedLabel.eq(target.data).cpu().sum().item() ) / float(target.size(0) )

        return loss, accuracy


class faceNet(nn.Module):
    def __init__(self,classnum=10574, feature=False, m = 1.35):
        super(faceNet, self).__init__()
        self.classnum = classnum
        self.feature = feature

        # IMPLEMENT resdiual network 20-layer with batch normalization

        self.fc6 = CustomLinear(in_features = 512,
                out_features = self.classnum, m=m)


    def forward(self, x):

        # IMPLEMENT forward of network


        if self.feature:
            return x

        x = self.fc6(x)
        return x
