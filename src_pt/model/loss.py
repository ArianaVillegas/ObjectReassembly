import torch
import torch.nn.functional as F
from torch.autograd import Variable

def contrastive_loss(y1, y2, gold, margin=1.0):
    euc_dist = F.pairwise_distance(y1, y2)

    loss = 0
    size = gold.shape[0]
    for i in range(size):
        label = gold[i]
        if torch.is_nonzero(label):
            loss += torch.pow(euc_dist[i], 2)
        else:
            delta = margin - euc_dist[i]
            delta = torch.clamp(delta, min=0.0, max=None)
            loss += torch.pow(delta, 2)
      
    loss = Variable((loss / size).clone().detach().requires_grad_(True), requires_grad=True)

    return loss