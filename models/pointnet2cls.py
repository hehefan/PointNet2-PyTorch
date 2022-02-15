import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

from pointnet2_modules import *

class Model(nn.Module):
    def __init__(self, input_feature_dim=0, num_classes=40):
        super(Model, self).__init__()

        self.input_feature_dim = input_feature_dim

        self.pointnet_sa_module1 = PointnetSAModule(mlp=[input_feature_dim, 64,  64,  128], npoint= 512, radius= 0.2, nsample=  32, bn=True, use_xyz=True)
        self.pointnet_sa_module2 = PointnetSAModule(mlp=[128,              128, 128,  256], npoint= 128, radius= 0.4, nsample=  64, bn=True, use_xyz=True)
        self.pointnet_sa_module3 = PointnetSAModule(mlp=[256,              256, 512, 1024], npoint=None, radius=None, nsample=None, bn=True, use_xyz=True)

        self.classifier = torch.nn.Sequential(torch.nn.Linear(in_features=1024, out_features=512),
                                              torch.nn.ReLU(inplace=True),
                                              torch.nn.Dropout(p=0.5),
                                              torch.nn.Linear(in_features=512, out_features=256),
                                              torch.nn.ReLU(inplace=True),
                                              torch.nn.Dropout(p=0.5),
                                              torch.nn.Linear(in_features=256, out_features=num_classes))

    def forward(self, input):
        if self.input_feature_dim == 0:
            xyz = input[:,:,:3].contiguous()
            feat = None
        else:
            xyz = input[:,:,:3].contiguous()
            feat = input[:,:,3:].permute(0, 2, 1).contiguous()

        xyz1, feat1 = self.pointnet_sa_module1(xyz, feat)
        xyz2, feat2 = self.pointnet_sa_module2(xyz1, feat1)
        _, feat3 = self.pointnet_sa_module3(xyz2, feat2)

        feat3 = torch.squeeze(input=feat3, dim=-1)

        output = self.classifier(feat3)

        return output
