import os
import glob
import h5py
import numpy as np

from torch.utils.data import Dataset
from src.dataloader.utils import transform


def load_data(partition, data_dir):
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(data_dir, 'modelnet40', 'ply_data_%s*.h5'%partition)):
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

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = transform(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]