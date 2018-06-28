# xyz
# Jan 2018
import os, sys
import numpy as np
from plyfile import PlyData, PlyElement
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR+'/all_datasets_meta')
from datasets_meta import DatasetsMeta

def test_plyfile():
    vertex = np.array([(0, 0, 0), (0,1,0), (1,1,0), (1,0,0),
                        (0, 0, 1), (0,1,1), (1,1,1), (1,0,1)],
                        dtype=[('x', 'f8'), ('y', 'f8'),
                               ('z', 'f8')])
    el_vertex = PlyElement.describe(vertex,'vertex')

    #PlyData([el]).write('tmp/test_binary.ply')
    #PlyData([el],text=True).write('tmp/test_ascii.ply')

    face = np.array([([0, 1, 2], 255, 255, 255),
                     ([0, 2, 3], 255,   0,   0),
                     ([0, 1, 3],   0, 255,   0),
                     ([1, 2, 3],   0,   0, 255)],
                    dtype=[('vertex_indices', 'i4', (3,)),
                           ('red', 'u1'), ('green', 'u1'),
                           ('blue', 'u1')])
    el_face = PlyElement.describe(face,'face')

    edge = np.array([(0, 1, 255, 0, 0),
                     (1, 2, 255, 0, 0),
                     (2, 3, 255, 0, 0),
                     (3, 0, 255, 0, 0),
                     (0, 4, 255, 0, 0)],
                    dtype=[('vertex1', 'i4'), ('vertex2','i4'),
                           ('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    el_edge = PlyElement.describe(edge,'edge')

    PlyData([el_vertex,el_edge],text=True).write('tmp/test_ascii.ply')
   # PlyData([el],
   #         byte_order='>').write('tmp/big_endian_binary.ply')
    print('write tmp/test_ascii.ply')


def gen_box_pl( ply_fn, box_vertexes, pl_xyz=None ):
    '''
    box_vertexes:[num_box,8,3]
    pl_xyz:  [num_point,3]
    '''
    assert box_vertexes.ndim == 3
    assert box_vertexes.shape[1] == 8
    assert box_vertexes.shape[-1] == 3

    box_vertexes = np.reshape( box_vertexes, (-1,3) )
    num_box = box_vertexes.shape[0] // 8
    if type(pl_xyz) != type(None):
        assert pl_xyz.shape[-1] == 3
        pl_xyz = np.reshape( pl_xyz, (-1,3) )
        num_vertex = box_vertexes.shape[0] + pl_xyz.shape[0]
    else:
        num_vertex = box_vertexes.shape[0]
    vertex = np.zeros( shape=(num_vertex) ).astype([('x', 'f8'), ('y', 'f8'),('z', 'f8')])
    for i in range(box_vertexes.shape[0]):
        vertex[i] = ( box_vertexes[i,0],box_vertexes[i,1],box_vertexes[i,2] )

    if type(pl_xyz) != type(None):
        for i in range(pl_xyz.shape[0]):
            vertex[i+box_vertexes.shape[0]] = ( pl_xyz[i,0],pl_xyz[i,1],pl_xyz[i,2] )

    el_vertex = PlyElement.describe(vertex,'vertex')

    edge_basic = np.array([ (0, 1, 255, 0, 0),
                            (1, 2, 255, 0, 0),
                            (2, 3, 255, 0, 0),
                            (3, 0, 255, 0, 0),
                            (4, 5, 255, 0, 0),
                            (5, 6, 255, 0, 0),
                            (6, 7, 255, 0, 0),
                            (7, 4, 255, 0, 0),
                            (0, 4, 255, 0, 0),
                            (1, 5, 255, 0, 0),
                            (2, 6, 255, 0, 0),
                            (3, 7, 255, 0, 0)] )
    edge_basic[:,2:5] = np.array([0,0,255])
    edge_val = np.concatenate( [edge_basic]*num_box,0 )
    for i in range(num_box):
        edge_val[i*12:(i+1)*12,0:2] += (8*i)
    edge = np.zeros( shape=(edge_val.shape[0]) ).astype(
                    dtype=[('vertex1', 'i4'), ('vertex2','i4'),
                           ('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    for i in range(edge_val.shape[0]):
        edge[i] = ( edge_val[i,0], edge_val[i,1], edge_val[i,2], edge_val[i,3], edge_val[i,4] )
    el_edge = PlyElement.describe(edge,'edge')

    PlyData([el_vertex, el_edge],text=True).write(ply_fn)
    print('write %s ok'%(ply_fn))

def gen_box_8vertexs( bxyz_min, bxyz_max ):
    '''
    no rotation!!
    from block xyz_min and xyz_max, generate eight vertexs in order
    bxyz_min: [n_box,3]
    bxyz_max" [n_box,3]
    '''
    if bxyz_min.ndim == 1:
        bxyz_max = np.expand_dims( bxyz_max,0 )
        bxyz_min = np.expand_dims( bxyz_min,0 )

    dxyz = bxyz_max - bxyz_min
    dx = dxyz[:,0]
    dy = dxyz[:,1]
    dz = dxyz[:,2]
    if bxyz_min.ndim==1:
        bxyz_min = np.expand_dims( bxyz_min,0 )

    box_vertexes = np.expand_dims( bxyz_min,1 )
    box_vertexes = np.tile( box_vertexes,[1,8,1] )
    box_vertexes[:,1,1] += dy
    box_vertexes[:,2,0] += dx
    box_vertexes[:,2,1] += dy
    box_vertexes[:,3,0] += dx

    box_vertexes[:,4:8,:] = box_vertexes[:,0:4,:] + 0
    box_vertexes[:,4:8,2] += np.expand_dims( dz,-1 )
    return box_vertexes

def gen_box_norotation( ply_fn, bxyz_min, bxyz_max ):
    box_vertexes = gen_box_8vertexs( bxyz_min, bxyz_max )
    gen_box_pl( ply_fn, box_vertexes )

def test_box( pl_xyz=None ):
    box_vertex0_a = np.array( [(0,0,0),(0,1,0),(1,1,0),(1,0,0)] )
    box_vertex0_b = box_vertex0_a + np.array([0,0,1])
    box_vertex0 = np.concatenate([box_vertex0_a,box_vertex0_b],0)
    box_vertex0 = np.expand_dims( box_vertex0,0 )
    box_vertex1 = box_vertex0 + np.array([[0.3,0.3,0]])
    box_vertexes = np.concatenate( [box_vertex0,box_vertex1],0 )


    bxyz_min = np.array([[0,0,0], [0,3,4]])
    bxyz_max = np.array([[1,2,1], [1,4,6]])
    box_vertexes = gen_box_8vertexs( bxyz_min, bxyz_max )
    pl_xyz = box_vertexes + np.array([-0.2,-0.3,0.2])

    gen_box_pl( '/tmp/box_pl.ply',box_vertexes, pl_xyz )

def cut_xyz( xyz, cut_threshold = [1,1,1] ):
    xyz = np.reshape( xyz,[-1,xyz.shape[-1]] )
    if np.sum(cut_threshold) == 3:
        is_keep = np.array( [True]*xyz.shape[0] )
        return xyz, is_keep
    cut_threshold = np.array( cut_threshold )
    xyz_min = np.amin( xyz[:,0:3], 0 )
    xyz_max = np.amax( xyz[:,0:3], 0 )
    xyz_scope = xyz_max - xyz_min
    xyz_cut = xyz_min + xyz_scope * cut_threshold
    tmp = xyz[:,0:3] <= xyz_cut
    is_keep = np.array( [True]*xyz.shape[0] )
    for i in range(xyz.shape[0]):
        is_keep[i] = tmp[i].all()
    xyz_new = xyz[is_keep,:]
    return xyz_new, is_keep

def create_ply( xyz0, ply_fn, label=None, label2color=None, force_color=None, box=None, cut_threshold=[1,1,1] ):
    '''
    xyz0: (num_block,num_point,3) or (num_point,3) or (num_block,num_point,6) or (num_point,6)
    label: (num_block,num_point) or (num_point)
    force_color: (3,)

    (1) color in xyz0: xyz0.shape[-]==6,  label=None, label2color=None
    (2) no color: xyz0.shape[-1]==3, label=None, label2color=None
    (3) color by label: xyz0.shape[-1]==3,label.shape= (num_block,num_point) or (num_point)
    (4) set consistent color: xyz0.shape[-1]==3, label=None, label2color=None, force_color=[255,0,0]
  '''
    print(ply_fn)
    folder = os.path.dirname(ply_fn)
    if not os.path.exists(folder):
        os.makedirs(folder)
    new_xyz_ls = []
    is_keep_ls = []
    is_cut = xyz0.ndim >= 3
    if is_cut:
        for i in range( xyz0.shape[0] ):
            new_xyz_i, is_keep_i = cut_xyz( xyz0[i], cut_threshold )
            new_xyz_ls.append( new_xyz_i )
            is_keep_ls.append( is_keep_i )
        xyz = np.concatenate( new_xyz_ls, 0 )
        is_keep = np.concatenate( is_keep_ls, 0 )
    else:
        xyz = xyz0

    if xyz.shape[-1] == 3:
        if type(label) != type(None) and label2color!=None:
            #(3) color by label: xyz0.shape[-1]==3,label.shape= (num_block,num_point) or (num_point)
            label2color_ls = []
            for i in range(len(label2color)):
                label2color_ls.append( np.reshape(np.array(label2color[i]),(1,3)) )
            label2colors = np.concatenate( label2color_ls,0 )
            color = np.take( label2colors,label,axis=0 )
            color = np.reshape( color,(-1,3) )
            if is_cut:
                color = color[is_keep,:]
            xyz = np.concatenate([xyz,color],-1)
        elif type(force_color)!=type(None):
            #(4) set consistent color: xyz0.shape[-1]==3, label=None, label2color=None, force_color=[255,0,0]
            color = np.reshape( np.array( force_color ),[1,3] )
            color = np.tile( color,[xyz.shape[0],1] )
            xyz = np.concatenate([xyz,color],-1)
    if xyz.shape[-1] == 3:
        #(2) no color: xyz0.shape[-1]==3, label=None, label2color=None
        vertex = np.zeros( shape=(xyz.shape[0]) ).astype([('x', 'f8'), ('y', 'f8'),('z', 'f8')])
        for i in range(xyz.shape[0]):
            vertex[i] = ( xyz[i,0],xyz[i,1],xyz[i,2] )
    elif xyz.shape[-1] == 6:
        vertex = np.zeros( shape=(xyz.shape[0]) ).astype([('x', 'f8'), ('y', 'f8'),('z', 'f8'),('red','u1'),('green','u1'),('blue','u1')])
        for i in range(xyz.shape[0]):
            vertex[i] = ( xyz[i,0],xyz[i,1],xyz[i,2],xyz[i,3],xyz[i,4],xyz[i,5] )
    else:
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        pass

    el_vertex = PlyElement.describe(vertex,'vertex')
    PlyData([el_vertex],text=True).write(ply_fn)

    print('save ply file: %s'%(ply_fn))


def create_ply_dset( dataset_name, xyz, ply_fn, label=None, cut_threshold=[1,1,1] ):
    create_ply( xyz, ply_fn, label = label, label2color = DatasetsMeta.g_label2color[dataset_name], cut_threshold=cut_threshold )

if __name__ == '__main__':
    #test_plyfile()
    test_box()

    #sg_bidxmap_i0 = np.arange( sg_bidxmap_i1.shape[0] ).reshape([-1,1,1,1])
    #sg_bidxmap_i0 = np.tile( sg_bidxmap_i0, [0,sg_bidxmap_i1.shape[1], sg_bidxmap_i1.shape[2],1] )
