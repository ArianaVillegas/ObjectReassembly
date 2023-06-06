"""
vn_dgcnn.py
The file has the original implementation of VN DGCCN and our proposed reduced version.

This code includes parts copied from vn_dgcnn_cls.py and vn_dgcnn_util.py

Original Code Repository: https://github.com/FlyingGiraffe/vnn/blob/master/models/vn_dgcnn_cls.py
                          https://github.com/FlyingGiraffe/vnn/blob/master/models/utils/vn_dgcnn_util.py

All rights and credit for the original code belong to the original author(s) and can be found in the original repository.
"""

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from src.model.extractor.vn_layers import VNLinearLeakyReLU, VNStdFeature, mean_pool


EPS = 1e-6


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def vn_get_graph_feature(x, k=20, idx=None, x_coord=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if x_coord is None:
            idx = knn(x, k=k)
        else:
            idx = knn(x_coord, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3) 
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous()

    return feature


class VN_DGCNN_ORIG(nn.Module):
    def __init__(self, args, num_class=40, normal_channel=False):
        super(VN_DGCNN_ORIG, self).__init__()
        self.args = args
        self.k = args.k
        
        self.conv1 = VNLinearLeakyReLU(6//3, 64//3)
        self.conv2 = VNLinearLeakyReLU(64//3*2, 64//3)
        self.conv3 = VNLinearLeakyReLU(64//3*2, 128//3)
        self.conv4 = VNLinearLeakyReLU(128//3*2, 256//3)

        self.conv5 = VNLinearLeakyReLU(256//3+128//3+64//3*2, 1024//3, dim=4, share_nonlinearity=True)
        
        self.std_feature = VNStdFeature(1024//3*2, dim=4, normalize_frame=False)
        self.linear1 = nn.Linear((1024//3)*12, 512)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, num_class)
        
        self.pool1 = mean_pool
        self.pool2 = mean_pool
        self.pool3 = mean_pool
        self.pool4 = mean_pool

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1)
        x = vn_get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = self.pool1(x)
        
        x = vn_get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = self.pool2(x)
        
        x = vn_get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = self.pool3(x)
        
        x = vn_get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = self.pool4(x)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        
        num_points = x.size(-1)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, trans = self.std_feature(x)
        x = x.view(batch_size, -1, num_points)
        
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        
        x = F.leaky_relu(self.bn1(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn2(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        
        return x


class VN_DGCNN(nn.Module):
    def __init__(self, args, output_channels=40, normal_channel=False):
        super(VN_DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        
        self.conv1 = VNLinearLeakyReLU(6//3, 64//3)
        self.conv2 = VNLinearLeakyReLU(64//3*2, 64//3)
        self.conv3 = VNLinearLeakyReLU(64//3*2, 64//3)
        self.conv4 = VNLinearLeakyReLU(64//3*2, 64//3)
        self.conv5 = VNLinearLeakyReLU(64//3*4, 64//3)

        self.conv6 = VNLinearLeakyReLU(64//3, 256//3, dim=4, share_nonlinearity=True)
        self.std_feature = VNStdFeature(256//3*2, dim=4, normalize_frame=False)
        
        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(args.dropout),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(args.dropout),
            nn.Linear(64, output_channels)
        )
        
        self.pool1 = mean_pool
        self.pool2 = mean_pool
        self.pool3 = mean_pool
        self.pool4 = mean_pool
        self.pool5 = mean_pool

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1)
        x = vn_get_graph_feature(x, k=self.k)
        x1_ = self.conv1(x)
        x1 = self.pool1(x1_)
        
        x = vn_get_graph_feature(x1, k=self.k)
        x2_ = self.conv2(x)
        x2 = self.pool2(x2_)
        
        x = vn_get_graph_feature(x2, k=self.k)
        x3_ = self.conv3(x)
        x3 = self.pool3(x3_)
        
        x = vn_get_graph_feature(x3, k=self.k)
        x4_ = self.conv4(x)
        x4 = self.pool4(x4_)
        
        x5 = torch.cat((x1_, x2_, x3_, x4_), dim=1)
        
        x = self.conv5(x5)
        x = self.pool5(x)
        
        x = self.conv6(x5)
        num_points = x.size(-1)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, trans = self.std_feature(x)
        x = x.view(batch_size, -1, num_points)
        
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = self.mlp(x)
        
        return x
