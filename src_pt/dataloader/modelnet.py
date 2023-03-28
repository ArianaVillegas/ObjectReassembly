import os
import glob
import h5py
import torch
import numpy as np

from torch.utils.data import Dataset
from src_pt.utils.pcd import translate_pcd


def load_data(partition, data_dir):
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(data_dir, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


class ModelNet40(Dataset):
    def __init__(self, num_points, data_dir, partition='train'):
        self.data, self.label = load_data(partition, data_dir)
        self.num_points = num_points
        self.partition = partition    
        self.group_examples()

    def group_examples(self):
        np_unique = np.unique(self.label)
        self.grouped = {}
        for lab in np_unique:
            self.grouped[lab] = np.where((self.label==lab))[0]

    def __getitem__(self, item):
        left_class = np.random.choice(np.asarray(list(self.grouped.keys())))
        left_idx = np.random.choice(self.grouped[left_class])
        if item % 2 == 0:
            right_idx = np.random.choice(self.grouped[left_class])
            while left_idx == right_idx:
                right_idx = np.random.choice(self.grouped[left_class])
            target = torch.tensor(1, dtype=torch.float)
        else:
            right_class = np.random.choice(np.asarray(list(self.grouped.keys())))
            while left_class == right_class:
                right_class = np.random.choice(np.asarray(list(self.grouped.keys())))
            right_idx = np.random.choice(self.grouped[right_class])
            target = torch.tensor(0, dtype=torch.float)
        left_pcd = self.data[left_idx][:self.num_points]
        right_pcd = self.data[right_idx][:self.num_points]
        if self.partition == 'train':
            left_pcd = translate_pcd(left_pcd)
            np.random.shuffle(left_pcd)
            right_pcd = translate_pcd(right_pcd)
            np.random.shuffle(right_pcd)
        return left_pcd, right_pcd, target

    def __len__(self):
        return self.data.shape[0]