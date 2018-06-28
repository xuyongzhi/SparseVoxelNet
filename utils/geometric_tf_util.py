# May 2018 xyz
import numpy as np
import tensorflow as tf


def tf_Rx( x, eager=False ):
    # anticlockwise, x: radian
    x = tf.cast(x, tf.float32)
    Rx = tf.concat([[[  1.0,  0.0,        0.0       ],
                     [  0.0,  tf.cos(x),  tf.sin(x) ],
                     [  0.0,  -tf.sin(x), tf.cos(x) ]]],
                     axis=0)
    return Rx

def tf_Ry( y,eager=False ):
    # anticlockwise, y: radian
    y = tf.cast(y, tf.float32)
    Ry = tf.concat([[[  tf.cos(y),  0.0,  -tf.sin(y)],
                     [  0.0,        1.0,  0.0     ],
                     [  tf.sin(y),  0.0,  tf.cos(y)]]],
                     axis=0)
    return Ry

def tf_Rz( z, eager=False ):
    # anticlockwise, z: radian
    z = tf.cast(z, tf.float32)
    Rz = tf.concat([[[  tf.cos(z),  tf.sin(z),  0.0],
                      [ -tf.sin(z), tf.cos(z),  0.0],
                      [ 0.0,        0.0,        1.0]]],
                     axis=0)
    return Rz

def tf_R1D( angle, axis, eager=False ):
    if axis == 'x':
        return tf_Rx(angle, eager)
    elif axis == 'y':
        return tf_Ry(angle, eager)
    elif axis == 'z':
        return tf_Rz(angle, eager)
    else:
        raise NotImplementedError

def tf_EulerRotate( angles, order ='zxy', eager=False ):
  '''
    angles: (3) float, unit :rad
  '''
  R = tf.eye(3, dtype=tf.float32)
  for i in range(3):
      R_i = tf_R1D(angles[i], order[i], eager)
      R = tf.matmul( R_i, R )
  return R

def main_test_eager():
  tf.enable_eager_execution()
  angles = np.array([0.5,0.5,0.5], np.float32) * np.pi
  Rz = tf_Rz(angles[2], eager=True)
  print(Rz)

  angles = tf.random_uniform([3]) * 2 * np.pi
  R = tf_EulerRotate(angles, eager=True)
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  pass

def main_test():
  #angles = np.array([0.5,0.5,0.5], np.float32) * np.pi
  angles = tf.random_uniform([3]) * 2 * np.pi
  angles = np.array([1,1,1]) * np.pi
  Rz = tf_Rz(angles[2])
  R = tf_EulerRotate(angles)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    Rz, R = sess.run([Rz, R])
    R_check = np.matmul(R, R.transpose())
    print(Rz)
    print(R)
    print(R_check)
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  pass

if __name__ == '__main__':
  main_test()

