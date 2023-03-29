import torch
import torch.nn as nn
import torch.nn.functional as F

from src_pt.model.extractor import DGCNN, get_graph_feature

class SiameseNet(nn.Module):
    def __init__(self, args):
        super(SiameseNet, self).__init__()
        self.args = args
        self.extractor = DGCNN(args)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv_a = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.ReLU())
        self.conv_b = nn.Sequential(nn.Conv2d(128*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.ReLU())
        
        self.conv_skip = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.conv1 = nn.Conv1d(192, 128, kernel_size=1, bias=False)
        
        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def extract_features(self, x):
        output = self.extractor(x)
        return output
        
    def forward(self, x1, x2):
        output1 = self.extract_features(x1)
        output2 = self.extract_features(x2)
        output = torch.cat((output1, output2), dim=1)
        
        output = get_graph_feature(output, k=self.args.k)
        output = self.conv_a(output)
        output_a = output.max(dim=-1, keepdim=False)[0]
        
        output = get_graph_feature(output_a, k=self.args.k)
        output = self.conv_b(output)
        output_b = output.max(dim=-1, keepdim=False)[0]
        
        output = torch.cat((output_a, output_b), dim=1)
        
        output = self.conv1(output)
        
        skip_output = self.conv_skip(output_a)
        output = output + skip_output
        
        batch_size = output.shape[0]
        output3 = F.adaptive_max_pool1d(output, 1).view(batch_size, -1)
        output4 = F.adaptive_avg_pool1d(output, 1).view(batch_size, -1)
        output = torch.cat((output3, output4), dim=1)
        output = self.fc(output)
        
        return output 