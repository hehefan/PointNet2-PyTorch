import os
import sys
import numpy as np
import h5py

def shuffle_points(data):
    """ Shuffle orders of points in a point cloud -- changes FPS behavior.
        Use the same shuffling idx for the entire batch.
        Input:
            NxC array
        Output:
            NxC array
    """
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    return data[idx,:]

def rotate_point_cloud(data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point cloud
        Return:
          Nx3 array, rotated point cloud
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_data = np.dot(data.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def rotate_point_cloud_z(data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point cloud
        Return:
          Nx3 array, rotated point cloud
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, sinval, 0],
                                [-sinval, cosval, 0],
                                [0, 0, 1]])
    rotated_data = np.dot(data.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def rotate_point_cloud_with_normal(xyz_normal):
    ''' Randomly rotate XYZ, normal point cloud.
        Input:
            xyz_normal: N,6, first three channels are XYZ, last 3 all normal
        Output:
            N,6, rotated XYZ, normal point cloud
    '''
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    shape_pc = xyz_normal[:,0:3]
    shape_normal = xyz_normal[:,3:6]
    xyz_normal[:,0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    xyz_normal[:,3:6] = np.dot(shape_normal.reshape((-1, 3)), rotation_matrix)
    return xyz_normal

def rotate_perturbation_point_cloud_with_normal(data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          Nx6 array, original point cloud and point normals
        Return:
          Nx3 array, rotated point clouds
    """
    rotated_data = np.zeros(data.shape, dtype=np.float32)
    angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
    R = np.dot(Rz, np.dot(Ry,Rx))
    shape_pc = data[:,0:3]
    shape_normal = data[:,3:6]
    rotated_data[:,0:3] = np.dot(shape_pc.reshape((-1, 3)), R)
    rotated_data[:,3:6] = np.dot(shape_normal.reshape((-1, 3)), R)
    return rotated_data


def rotate_point_cloud_by_angle(data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          Nx3 array, original point cloud
        Return:
          Nx3 array, rotated point cloud
    """
    #rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_data = np.dot(data.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def rotate_point_cloud_by_angle_with_normal(data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx6 array, original point cloud with normal
          scalar, angle of rotation
        Return:
          BxNx6 array, rotated point cloud iwth normal
    """
    rotated_data = np.zeros(data.shape, dtype=np.float32)
    #rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    shape_pc = data[:,0:3]
    shape_normal = data[:,3:6]
    rotated_data[:,0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    rotated_data[:,3:6] = np.dot(shape_normal.reshape((-1,3)), rotation_matrix)
    return rotated_data


def rotate_perturbation_point_cloud(data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          Nx3 array, original point cloud
        Return:
          Nx3 array, rotated point cloud
    """
    angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1,0,0],
                   [0,np.cos(angles[0]),-np.sin(angles[0])],
                   [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                   [0,1,0],
                   [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                   [np.sin(angles[2]),np.cos(angles[2]),0],
                   [0,0,1]])
    R = np.dot(Rz, np.dot(Ry,Rx))
    rotated_data = np.dot(data.reshape((-1, 3)), R)
    return rotated_data


def jitter_point_cloud(data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          Nx3 array, original point cloud
        Return:
          Nx3 array, jittered point cloud
    """
    N, C = data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    jittered_data += data
    return jittered_data

def shift_point_cloud(data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          Nx3 array, original point cloud
        Return:
          Nx3 array, shifted point cloud
    """
    N, C = data.shape
    shift = np.random.uniform(-shift_range, shift_range, 3)
    shifted_data = data + shift
    return shifted_data


def random_scale_point_cloud(data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            Nx3 array, original point cloud
        Return:
            Nx3 array, scaled point cloud
    """
    N, C = data.shape
    scale = np.random.uniform(scale_low, scale_high)
    scaled_scale = data * scale
    return scaled_scale

def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' pc: Nx3 '''
    dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc


def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)
