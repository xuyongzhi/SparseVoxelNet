# xyz June 2018

import tensorflow as tf
import os, sys

BASE_DIR = os.path.abspath(__file__)
sys.path.append(BASE_DIR)

from geometric_tf_util import *
import numpy as np



def random_rotate(max_angles_yxz):
  '''
      Rotate the whole object by a random angle 3D
      unit: rad
  '''

  angles = tf.random_uniform([3], dtype=tf.float32) * 2 * np.pi
  angles = tf.minimum(angles, max_angles_yxz)
  R = tf_EulerRotate(angles, 'yxz')
  return R, angles

def rotate_perturbation(angle_sigma=0.06, angle_clip=0.18):
  '''
      Rotate the whole object by a random angle 3D
      unit: rad
   '''
  angles = tf.random_normal([3], mean=0, stddev=angle_sigma)
  p_angles = tf.clip_by_value(angles, -angle_clip, angle_clip)

  pR = tf_EulerRotate(angles, 'yxz')
  return pR, p_angles

def random_scaling(min_scale=0.8, max_scale=1.25):
  scales = tf.random_uniform([3], minval=min_scale, maxval=max_scale)
  S = tf.concat([[[scales[0], 0, 0],
                  [0, scales[0], 0],
                  [0, 0, scales[0]] ]], 0)
  return S

def random_shift(shift_range=0.1):
  shifts = tf.random_uniform([1,3], minval=-shift_range, maxval=shift_range)
  return shifts

def random_jitter(points_shape, sigma=0.01, clip=0.05):
  jitter = tf.random_normal(points_shape, mean=0.0, stddev=sigma)
  jitter = tf.clip_by_value(jitter, -clip, clip)
  return jitter

def aug_data(points, b_bottom_centers_mm, data_idxs, \
            aug_items, aug_metas={} ):
  '''
      points: [n,3] 'xyz'
              [n,6] 'xyznxnynz'
  '''
  assert len(points.shape) == 2
  point_num = points.shape[0].value
  channels = points.shape[-1].value
  cascade_num = len(b_bottom_centers_mm)
  for c in range(cascade_num):
    b_bottom_centers_mm[c] = tf.cast(tf.reshape(b_bottom_centers_mm[c], [-1,3]),
                                     tf.float32)

  assert (data_idxs['xyz'] == np.array([0,1,2])).all()
  if channels>3:
    assert (data_idxs['nxnynz'] == np.array([3,4,5])).all()
  assert channels==3 or channels==6

  augs = {}
  RS = tf.eye(3, dtype=tf.float32)
  if 'rotation' in aug_items:
    R, angles = random_rotate(aug_metas['max_angles_yxz'])
    RS = tf.matmul(RS, R)
    augs['R'] = R
    augs['angles_yxz'] = angles

  if 'perturbation' in aug_items:
    pR, p_angles = rotate_perturbation()
    RS = tf.matmul(RS, pR)
    augs['pR'] = pR
    augs['p_angles'] = p_angles

  if 'scaling' in aug_items:
    S = random_scaling()
    RS = tf.matmul(RS, S)
    augs['S'] = S
  augs['RS'] = RS

  points_xyz = tf.matmul(points[:,0:3], RS)
  for c in range(cascade_num):
    b_bottom_centers_mm[c] = tf.matmul(b_bottom_centers_mm[c], RS)


  if 'shifts' in aug_items:
    shifts = random_shift()
    points_xyz += shifts
    for c in range(cascade_num):
      b_bottom_centers_mm[c] += shifts
    augs['shifts'] = shifts

  if 'jitter' in aug_items:
    jitter = random_jitter((point_num,3))
    points_xyz += jitter
    augs['jitter'] = jitter

  if channels==3:
    points = points_xyz
  elif channels==6:
    if 'rotation' in aug_items:
      points_normal = tf.matmul(points[:,3:6], R)
    else:
      points_normal = points[:,3:6]
    points = tf.concat([points_xyz, points_normal], -1)

  for c in range(cascade_num):
    b_bottom_centers_mm[c] = tf.reshape(b_bottom_centers_mm[c], [-1,6])

  return points, b_bottom_centers_mm, augs

def parse_augtypes(aug_types):
  tmp = aug_types.split('-')
  for s in tmp[0]:
    assert s in 'Nrpsfj', ('%s not in Nrsfj'%(s))
  if 'N' in tmp[0]:
    assert tmp[0]=='N'
    aug_items=[]
  else:
    to_aug_items = {'r':'rotation', 'p':'perturbation', 's':'scaling', 'f':'shifts', 'j':'jitter'}
    aug_items = [to_aug_items[e] for e in tmp[0]]
  aug_metas = {}
  if len(tmp)>1:
    aug_metas['max_angles_yxz'] = np.array([float(a) for a in tmp[1].split('_')])*np.pi/180.0

  return aug_items, aug_metas

def aug_main(points, b_bottom_centers_mm, aug_types, data_idxs):
  if aug_types=='none':
    return points, b_bottom_centers_mm, None
  else:
    aug_items, aug_metas = parse_augtypes(aug_types)
    return aug_data(points, b_bottom_centers_mm, data_idxs, aug_items, aug_metas)


