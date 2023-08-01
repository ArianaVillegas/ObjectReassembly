import os
import re
import igl
import torch
import argparse
import numpy as np
from scipy.spatial import distance
from dgl.geometry import farthest_point_sampler


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(BASE_DIR, '../geometry')


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def get_pieces(folder):
    vs = []
    fs = []
    labels = []
    dirs = os.listdir(folder)
    dirs.sort(key=natural_keys)
    for filename in dirs:
        v, f = igl.read_triangle_mesh(os.path.join(folder, filename))
        vs.append(v)
        fs.append(f)
        labels.append(filename)
    return vs, fs, labels


def find_common(left_face, right_face, dist):
    left_idx = []
    right_idx = []
    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            if dist[i][j] == 0:
                left_idx.append(i)
                right_idx.append(j)
    left_sub_face = [sum(np.in1d(face, left_idx)) for face in left_face]
    right_sub_face = [sum(np.in1d(face, right_idx)) for face in right_face]
    return (max(left_sub_face) == 3) & (max(right_sub_face) == 3)


def get_pairs(vs, fs, labels):
    size = len(vs)
    pairs = []
    for i in range(size-1):
        for j in range(i+1, size):
            dist = distance.cdist(vs[i], vs[j])
            face = find_common(fs[i], fs[j], dist)
            if face:
                pairs.append((labels[i], labels[j]))
    return pairs



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Feature extractor')
    parser.add_argument('--subset', default='artifact', type=str)
    args = parser.parse_args()
    subset = args.subset

    root_folder = f'{data_dir}/{subset}'
    label_folder = f'{data_dir}/labels_matching_{subset}'
    os.makedirs(label_folder, exist_ok=True)

    vs_fl = []
    label_fl = []
    for folder in os.listdir(root_folder):
        if folder.endswith('.py') or folder.endswith('.txt'):
            continue
        cur = os.path.join(root_folder, folder)
        lcur = os.path.join(label_folder, folder)
        os.makedirs(lcur, exist_ok=True)
        list_cur = os.listdir(cur)
        
        for cur_folder in list_cur:
            if cur_folder == 'nsm' or cur_folder == 'original':
                print('Avoid', os.path.join(cur, cur_folder))
                continue
            vs, fs, labels = get_pieces(os.path.join(cur, cur_folder))
            pairs = get_pairs(vs, fs, labels)
            print(f'{os.path.join(lcur, cur_folder)}.txt')
            with open(f'{os.path.join(lcur, cur_folder)}.txt', 'w') as f:
                lp = len(pairs)
                size = (len(labels)**2 - len(labels) - 2*lp) // 2
                f.write(f'{lp} {size}\n')
                
                for pair in pairs:
                    f.write(f'{pair[0]} {pair[1]} 1\n')
                for i in range(len(labels)):
                    for j in range(i, len(labels)):
                        if i != j and (labels[i], labels[j]) not in pairs:
                            f.write(f'{labels[i]} {labels[j]} 0\n')

            labels = [os.path.join(folder, cur_folder, label) for label in labels]
            print('NEW LABELS')
            print(labels)
            for v in vs:
                v = np.array(v)
                if v.shape[0] < 2048:
                    idx = np.random.choice(v.shape[0], 2048-v.shape[0])
                    v = np.concatenate((v, v[idx]), axis=0)
                v_idx = farthest_point_sampler(torch.unsqueeze(torch.from_numpy(v), 0), 2048)
                vs_fl += [v[v_idx]]
            print(len(vs))
            label_fl += labels

    import pandas as pd
    df = pd.DataFrame({'label':label_fl, 'points':vs_fl})
    df['points'] = df['points'].apply(np.array)
    df.to_pickle(f'pcd_{subset}.pkl')