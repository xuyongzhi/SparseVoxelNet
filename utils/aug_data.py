# May 2018
# xyz

import os
import sys
import numpy as np
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

#import geometric_util as geo_util
import provider

def aug_id_to_type( aug_id ):
    aug_types = {}
    aug_types['RotateIn'] = False
    aug_types['RotateVox'] = False
    aug_types['RotateInXYZMax'] = np.array([15,15,360], dtype=np.float32)*np.pi/180.0

    aug_types['RotateVoxXYZChoices'] = [
                            np.array([], dtype=np.float32 ) * np.pi * 0.5,
                            np.array([], dtype=np.float32 ) * np.pi * 0.5,
                            np.array([-2,0,2], dtype=np.float32 ) * np.pi * 0.5 ]
    aug_types['RotateVoxXYZChoices'] = [
                            np.array([], dtype=np.float32 ) * np.pi * 0.5,
                            np.array([], dtype=np.float32 ) * np.pi * 0.5,
                            np.array([-3,-2,-1,0,1,2,3], dtype=np.float32 ) * np.pi * 0.5 ]
    if aug_id == 0:
        pass
    elif aug_id == 1:
        aug_types['RotateIn'] = True
    elif aug_id == 2:
        aug_types['RotateVox'] = True
    elif aug_id == 3:
        aug_types['RotateIn'] = True
        aug_types['RotateVox'] = True
    else:
        raise NotImplementedError
    return aug_types

def aug_batch( data, is_include_normal, aug_types, dataset_name  ):
    if 'RotateIn' not in aug_types:
        return data
    if dataset_name == 'MODELNET40':
        data = augment_batch_data_MODELNET( data, is_include_normal )
    else:
        assert NotImplementedError
    return data

def augment_batch_data_MODELNET( batch_data, is_include_normal ):
    '''
    is_include_normal=False: xyz
    is_include_normal=True: xyznxnynz
    '''
    if is_include_normal:
        rotated_data = provider.rotate_point_cloud_with_normal(batch_data)
        rotated_data = provider.rotate_perturbation_point_cloud_with_normal(rotated_data)
    else:
        rotated_data = provider.rotate_point_cloud(batch_data)
        rotated_data = provider.rotate_perturbation_point_cloud(rotated_data)

    jittered_data = provider.random_scale_point_cloud(rotated_data[:,:,0:3])
    jittered_data = provider.shift_point_cloud(jittered_data)
    jittered_data = provider.jitter_point_cloud(jittered_data)
    rotated_data[:,:,0:3] = jittered_data
    return provider.shuffle_points(rotated_data)

