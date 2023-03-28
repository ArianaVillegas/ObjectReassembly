import torch
import torch.nn as nn
import torch.nn.functional as F

from src_pt.model.extractor import DGCNN, get_graph_feature

class SiameseNet(nn.Module):
    def __init__(self, args):
        super(SiameseNet, self).__init__()
        self.args = args
        self.extractor = DGCNN(args)
        # self.extractor.linearf = nn.Linear(256, self.in_features)
        # self.conv = nn.Conv2d(args.emb_dims, 512, kernel_size=1, bias=False)
        self.conv_a = nn.Conv2d(256, 128, kernel_size=1, bias=False)
        self.conv_b = nn.Conv2d(128*2, 64, kernel_size=1, bias=False)
        self.conv_skip = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        
        self.conv1 = nn.Conv1d(192, 128, kernel_size=1, bias=False)
        # self.conv1 = nn.Conv1d(192, 256, kernel_size=1, bias=False)
        
        self.fc = nn.Sequential(
            # nn.Linear(512 * 2, 512),
            # nn.Linear(args.emb_dims * 4, 512),
            # nn.BatchNorm1d(512),
            # nn.LeakyReLU(inplace=True),
            # nn.Dropout(args.dropout),
            # nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            # nn.LeakyReLU(inplace=True),
            # nn.Dropout(args.dropout),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
            # nn.BatchNorm1d(32),
            # nn.LeakyReLU(0.01),
            # nn.Dropout(0.5),
            # nn.Linear(32, 1)
        )

        # self.sigmoid = nn.Sigmoid()
        
    def extract_features(self, x):
        output = self.extractor(x)
        # output = output.view(output.size()[0], -1)
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
        
        # Add skip connection
        skip_output = self.conv_skip(output_a)
        output = output + skip_output
        
        batch_size = output.shape[0]
        output3 = F.adaptive_max_pool1d(output, 1).view(batch_size, -1)
        output4 = F.adaptive_avg_pool1d(output, 1).view(batch_size, -1)
        output = torch.cat((output3, output4), dim=1)
        output = self.fc(output)

        # output = self.sigmoid(output)
        
        return output 