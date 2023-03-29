import pandas as pd
import numpy as np
import torch
import os

from torch.utils.data import Dataset
from src_pt.utils.pcd import translate_pcd

all_data = None
num_points = None



def load_data(data_dir):
    global all_data
    global num_points
    if all_data is None:
        df = pd.read_pickle(os.path.join(data_dir, 'pcd.pkl'))
        print(np.array(df['points'][df.index[0]]).shape)
        num_points = np.array(df['points'][df.index[0]]).shape[1]
        all_data = {row['label']: row['points'].reshape(num_points, 3) for _, row in df.iterrows()}
        print(df.head())
    print('num points', num_points)
    return all_data, num_points



def load_pairs(data_dir, partition, args):
    list_dir = os.path.join(data_dir, 'labels')
    all_both = []
    cnt = 0
    for folder in os.listdir(list_dir):
        if folder not in partition:
            continue
        obj_folder = os.path.join(list_dir, folder)
        obj_folder_list = os.listdir(obj_folder)
        np.random.shuffle(obj_folder_list)
        n_cuts = round(len(obj_folder_list) * args.prop_dt)
        for filename in obj_folder_list[:n_cuts]:
            filepath = os.path.join(obj_folder, filename)
            with open(filepath, 'r') as f:
                lines = f.readlines()
                ns, nd = map(int, lines[0].split())
                total = round(ns/args.prop_pn)
                snd = min(nd, round(total * (1-args.prop_pn)))
                cnt += ns + snd
                for i in range(ns):
                    l, r, cmp = lines[i+1].split()
                    l = os.path.join(folder, filename[:-4], l)
                    r = os.path.join(folder, filename[:-4], r)
                    cmp = int(cmp)
                    assert(cmp==1)
                    all_both.append((cmp, l, r))
                for i in np.random.choice(nd, snd, replace=False):
                    l, r, cmp = lines[i+1+ns].split()
                    l = os.path.join(folder, filename[:-4], l)
                    r = os.path.join(folder, filename[:-4], r)
                    cmp = int(cmp)
                    assert(cmp==0)
                    all_both.append((cmp, l, r))
    print(f'Read data {cnt} items')
    np.random.shuffle(all_both)
    return all_both


class BreakingBad(Dataset):
    def __init__(self, partition, data_dir, args, mode='train'):
        self.data, self.num_points = load_data(data_dir)
        self.both = load_pairs(data_dir, partition, args=args)
        self.mode = mode
        self.args = args

    def __getitem__(self, item):
        (cmp, left, right) = self.both[item]
        target = torch.tensor(cmp, dtype=torch.float)
        left_pcd = self.data[left][:self.num_points]
        right_pcd = self.data[right][:self.num_points]
        if self.mode == 'train':
            left_pcd = translate_pcd(left_pcd)
            np.random.shuffle(left_pcd)
            right_pcd = translate_pcd(right_pcd)
            np.random.shuffle(right_pcd)
        return left_pcd, right_pcd, target

    def __len__(self):
        return len(self.both)
    
    def get_num_points(self):
        return self.num_points
