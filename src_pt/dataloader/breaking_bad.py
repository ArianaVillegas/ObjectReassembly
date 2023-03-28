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


# all_same = None
# all_diff = None
# def load_pairs(data_dir, partition, prop=0.5):
#     global all_same
#     global all_diff
#     if all_same is None and all_diff is None:
#         df = pd.read_csv(os.path.join(data_dir, 'labels.txt'), sep=' ', names=['left', 'right', 'sim'], header=None, index_col=None)
#         df = df.sample(frac=1)
#         all_same = []
#         all_diff = []
#         all_tg = []
#         all_data = []
#         for ind in df.index:
#             left = df['left'][ind]
#             right = df['right'][ind]
#             sim = int(df['sim'][ind])
#             all_tg.append(sim)
#             all_data.append((left, right))
#             if sim:
#                 all_same.append((left, right))
#             else:
#                 all_diff.append((left, right))
#         all_same = np.array(all_same)
#         all_diff = np.array(all_diff)
#         np.random.shuffle(all_same)
#         np.random.shuffle(all_diff)
#     return all_same, all_diff

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
        # print(len(obj_folder_list), args.prop_dt)
        # print(n_cuts)
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
        # self.same, self.diff = load_pairs(data_dir, partition)
        self.both = load_pairs(data_dir, partition, args=args)
        self.mode = mode
        self.args = args
        
        # Select partition
        # self.same = self.same[partition]
        # self.diff = self.diff[partition]
        # self.both = self.both[partition]

    def __getitem__(self, item):
        (cmp, left, right) = self.both[item]
        target = torch.tensor(cmp, dtype=torch.float) 
        # if item % 2 == 0:
        #     (left, right) = self.diff[item//2]
        #     target = torch.tensor(0, dtype=torch.float) 
        # else:
        #     (left, right) = self.same[item//2]
        #     target = torch.tensor(1, dtype=torch.float) 
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
        # return self.same.shape[0] + self.diff.shape[0]
    
    def get_num_points(self):
        return self.num_points
    
# if __name__ == "__main__":
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     DATA_DIR = os.path.join(BASE_DIR, '../../data')
#     same, diff, tg, both = load_pairs(DATA_DIR)
#     print('Same shape', same.shape)
#     print('Diff shape', diff.shape)
    
    
    
# df = pd.read_csv(os.path.join(DATA_DIR, 'pcd.csv'), index_col='label')
# # df = df.head()
# df['label'] = df.index
# df['points'] = df['points'].apply(lambda x: np.matrix(x))
# df['size'] = df['points'].apply(lambda x: x.shape[0])
# size = np.max(df['size'])
# for i, row in df.iterrows():
#     # extra_idx = np.random.choice(row['size'], size - row['size'])
#     # extra = row['points'][extra_idx]
#     # df.at[i,'points'] = np.concatenate((row['points'], extra), axis=0)
#     extra_idx = np.random.choice(row['size'], num_points)
#     extra = row['points'][extra_idx]
#     df.at[i,'points'] = extra
# df['new_size'] = df['points'].apply(lambda x: x.shape)
# print(df[['size', 'new_size']].head())
# df[['label', 'points']].to_pickle(os.path.join(DATA_DIR, 'pcd_sample.pkl'))

# all_data = {}
# size = 0
# for ind in df.index:
#     cmp_name = df['label'][ind]
#     vs = df['points'][ind]
#     vs = np.matrix(vs)
#     size = max(size, vs.shape[0])
#     all_data[cmp_name] = vs
# for key in range(all_data):
#     cur_size = all_data[key].shape[0]
#     print('Name', key, 'Before', all_data[key].shape, end=' ')
#     extra_pts = np.random.choice(cur_size, size - cur_size)
#     extra_pts = all_data[key][extra_pts]
#     all_data[key] = np.concatenate((all_data[key], extra_pts), axis=0)
#     print('After', all_data[key].shape)
# return all_data, size