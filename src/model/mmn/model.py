import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.extractor import get_model, get_graph_feature, vn_get_graph_feature

class MatchMakerNet(nn.Module):
    def __init__(self, args):
        super(MatchMakerNet, self).__init__()
        self.args = args
        self.graph_ftr = None
        self.extractor = None
        self._get_model()
        
        self.conv_a = nn.Conv2d(256, 128, kernel_size=1, bias=False)
        self.conv_b = nn.Conv2d(128*2, 64, kernel_size=1, bias=False)
        self.conv_skip = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.conv1 = nn.Conv1d(192, 128, kernel_size=1, bias=False)
        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def _get_model(self):
        if self.args.extractor.endswith('ours'):
            self.extractor = get_model(self.args)
            if self.args.extractor == 'ours':
                self.graph_ftr = get_graph_feature
                
            elif self.args.extractor == 'vn_ours':
                self.graph_ftr = vn_get_graph_feature
        else:
            raise Exception("Not implemented other feature extractors in MatchMakerNet")
        
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