from scipy.spatial.transform import Rotation as R
import numpy as np


def transform(pcd):
    # Center pointcloud
    centroid = np.mean(pcd, axis=0)
    pcd = pcd - centroid[None]
    # Rotate pointcloud
    # rot_mat = R.random().as_matrix()
    # pcd = (rot_mat @ pcd.T).T
    # Translate pointcloud
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])  
    pcd = np.add(np.multiply(pcd, xyz1), xyz2)
    # Suffle pointcloud
    pcd = pcd.astype('float32')
    np.random.shuffle(pcd)
    return pcd   