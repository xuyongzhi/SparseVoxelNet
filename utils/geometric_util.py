# May 2018 xyz
import numpy as np


def Rx( x ):
    # ref to my master notes 2015
    # anticlockwise, x: radian
    Rx = np.zeros((3,3))
    Rx[0,0] = 1
    Rx[1,1] = np.cos(x)
    Rx[1,2] = np.sin(x)
    Rx[2,1] = -np.sin(x)
    Rx[2,2] = np.cos(x)
    return Rx

def Ry( y ):
    # anticlockwise, y: radian
    Ry = np.zeros((3,3))
    Ry[0,0] = np.cos(y)
    Ry[0,2] = -np.sin(y)
    Ry[1,1] = 1
    Ry[2,0] = np.sin(y)
    Ry[2,2] = np.cos(y)
    return Ry

def Rz( z ):
    # anticlockwise, z: radian
    Rz = np.zeros((3,3))

    Rz[0,0] = np.cos(z)
    Rz[0,1] = np.sin(z)
    Rz[1,0] = -np.sin(z)
    Rz[1,1] = np.cos(z)
    Rz[2,2] = 1
    return Rz

def R1D( angle, axis ):
    if axis == 'x':
        return Rx(angle)
    elif axis == 'y':
        return Ry(angle)
    elif axis == 'z':
        return Rz(angle)
    else:
        raise NotImplementedError

def EulerRotate( angles, order ='zxy' ):
    R = np.eye(3)
    for i in range(3):
        R_i = R1D(angles[i], order[i])
        R = np.matmul( R_i, R )
    return R

def point_rotation_randomly( points, rxyz_max=np.pi*np.array([0.1,0.1,0.1]) ):
    # Input:
    #   points: (B, N, 3)
    #   rx/y/z: in radians
    # Output:
    #   points: (B, N, 3)
    batch_size = points.shape[0]
    for b in range(batch_size):
        rxyz = [ np.random.uniform(-r_max, r_max) for r_max in rxyz_max ]
        R = EulerRotate( rxyz, 'xyz' )
        points[b,:,:] = np.matmul( points[b,:,:], np.transpose(R) )
    return points
