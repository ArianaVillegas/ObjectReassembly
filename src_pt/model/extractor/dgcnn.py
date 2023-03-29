import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k=20):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  
    
    device = x.get_device() # torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40) -> None:
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        # self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.ReLU())
                                   # nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.ReLU())
                                   # nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.ReLU())
                                   # nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.ReLU())
                                   # nn.LeakyReLU(negative_slope=0.2))
        
        self.conv_skip = nn.Conv2d(64*4, 64, kernel_size=1, bias=False)  # skip connection layer
        # self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
        #                            self.bn4,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
        #                            self.bn5,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.linear1 = nn.Linear(args.emb_dims*2, args.emb_dims, bias=False)
        # self.bn6 = nn.BatchNorm1d(args.emb_dims)
        # self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        # self.bn6 = nn.BatchNorm1d(512)
        # self.dp1 = nn.Dropout(p=args.dropout)
        # self.linear2 = nn.Linear(512, 256)
        # self.bn7 = nn.BatchNorm1d(256)
        # self.dp2 = nn.Dropout(p=args.dropout)
        # self.linearf = nn.Linear(256, output_channels)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        batch_size = x.shape[0]
        x = get_graph_feature(x, k=self.k)
        x1_ = self.conv1(x)
        x1 = x1_.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x2_ = self.conv2(x)
        x2 = x2_.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x3_ = self.conv3(x)
        x3 = x3_.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x4_ = self.conv4(x)
        x4 = x4_.max(dim=-1, keepdim=False)[0]

        # x = torch.cat((x1, x2, x3, x4), dim=1)
        x = torch.cat((x1_, x2_, x3_, x4_), dim=1)
        
        # apply the skip connection layer
        x = self.conv_skip(x)
        x = x.max(dim=-1, keepdim=False)[0]

        # x = self.conv5(x)
        # x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        # x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        # x = torch.cat((x1, x2), 1)

        # x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        # x = self.dp1(x)
        # x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        # x = self.dp2(x)
        # x = self.linearf(x)
        return x