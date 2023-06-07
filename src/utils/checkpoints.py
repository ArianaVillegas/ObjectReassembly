from __future__ import print_function
import torch
import os


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()
        
        
class SaveBestModel:
    def __init__(self, dir, metric=float('inf'), name='', idx=0, asc=False):
        self.dir = dir
        self.metric = -metric if asc else metric
        self.name = name
        self.idx = idx
        self.asc = asc
        
    def __call__(self, curr_metric, epoch, model, optimizer, criterion):
        if self.asc:
            improved = (curr_metric > self.metric)
        else:
            improved = (curr_metric < self.metric)
        if improved:
            self.metric = curr_metric
            print(f"\nBest validation: {self.metric}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f'{self.dir}/checkpoints/{self.name}/train/best_model_{self.idx}.pth')


def checkpoint_init(exp_name):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+exp_name):
        os.makedirs('checkpoints/'+exp_name)
    if not os.path.exists('checkpoints/'+exp_name+'/train'):
        os.makedirs('checkpoints/'+exp_name+'/train')
    if not os.path.exists('checkpoints/'+exp_name+'/test'):
        os.makedirs('checkpoints/'+exp_name+'/test')