import os
import sys
import numpy as np
import torch
from .transforms import *

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, npoints = 1024, normalize=True, normal_channel=False, modelnet10=False, train=True, memory=True):
        super(Dataset, self).__init__()

        self.root = root
        self.npoints = npoints
        self.normalize = normalize
        self.train = train
        if modelnet10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))  
        self.normal_channel = normal_channel

        shape_ids = {}
        if modelnet10:
            if train:
                shape_ids = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            else:
                shape_ids = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            if train:
                shape_ids = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            else:
                shape_ids = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[i])+'.txt') for i in range(len(shape_ids))]

        self.num_classes = len(self.cat)

        if memory:
            self.data = []
            for fn in self.datapath:
                self.data.append(np.loadtxt(fn[1],delimiter=',').astype(np.float32))
        self.memory = memory


    def __len__(self):
        return len(self.datapath)


    def _augment_batch_data(self, data):
        if self.normal_channel:
            rotated_data = rotate_point_cloud_with_normal(data)
            rotated_data = rotate_perturbation_point_cloud_with_normal(rotated_data)
        else:
            rotated_data = rotate_point_cloud(data)
            rotated_data = rotate_perturbation_point_cloud(data)

        jittered_data = random_scale_point_cloud(rotated_data[:,0:3])
        jittered_data = shift_point_cloud(jittered_data)
        jittered_data = jitter_point_cloud(jittered_data)
        rotated_data[:,0:3] = jittered_data
        return shuffle_points(rotated_data)

    def __getitem__(self, idx):

        if self.memory:
            point_set = self.data[idx]
        else:
            fn = self.datapath[idx]
            point_set = np.loadtxt(fn[1],delimiter=',').astype(np.float32)

        cls = self.classes[self.datapath[idx][0]]
        # Take the first npoints
        point_set = point_set[0:self.npoints,:]
        if self.normalize:
            point_set[:,0:3] = pc_normalize(point_set[:,0:3])
        if not self.normal_channel:
            point_set = point_set[:,0:3]
            if self.train:
                point_set = self._augment_batch_data(point_set)
            return point_set.astype(np.float32), cls
        else:
            if self.train:
                point_set = self._augment_batch_data(point_set)
            return point_set.astype(np.float32), cls

if __name__ == '__main__':
    dataset = Dataset(root='../data/modelnet40_normal_resampled')
    pc, label = dataset[0]
    print(pc.shape)
    print(label)
