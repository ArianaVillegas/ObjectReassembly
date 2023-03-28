import numpy as np

def translate_pcd(pcd):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pcd = np.add(np.multiply(pcd, xyz1), xyz2).astype('float32')
    return translated_pcd

def jitter_pcd(pcd, sigma=0.01, clip=0.02):
    N, C = pcd.shape
    pcd += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pcd