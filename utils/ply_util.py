# xyz
# Jan 2018
import os, sys
import numpy as np
from plyfile import PlyData, PlyElement
from datasets.all_datasets_meta.datasets_meta import DatasetsMeta

def gen_box_and_pl( ply_fn, box_vertexes, pl_xyz=None, extra='' ):
    '''
    Generate box and points together in the same file. box_vertexes and pl_xyz
      independently. The 8 vertexs are included automatically in the box.
      pl_xyz is used to provide other points.
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

    # define the order of the 8 vertexs for a box
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
    if extra=='random_color_between_boxes':
      color = np.random.randint(0,256,3)
      color[2] = 255 - color[0] - color[1]
    edge_basic[:,2:5] = color

    edge_val = np.concatenate( [edge_basic]*num_box,0 )
    for i in range(num_box):
        edge_val[i*12:(i+1)*12,0:2] += (8*i)
    edge = np.zeros( shape=(edge_val.shape[0]) ).astype(
                    dtype=[('vertex1', 'i4'), ('vertex2','i4'),
                           ('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    for i in range(edge_val.shape[0]):
        edge[i] = ( edge_val[i,0], edge_val[i,1], edge_val[i,2], edge_val[i,3], edge_val[i,4] )
    el_edge = PlyElement.describe(edge,'edge')

    dirname = os.path.dirname(ply_fn)
    if not os.path.exists(dirname):
      os.makedirs(dirname)
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
    gen_box_and_pl( ply_fn, box_vertexes )

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
    pl_xyz = None

    gen_box_and_pl( '/tmp/box_pl.ply',box_vertexes, pl_xyz )

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
      raise NotImplementedError

    el_vertex = PlyElement.describe(vertex,'vertex')
    PlyData([el_vertex],text=True).write(ply_fn)

    print('save ply file: %s'%(ply_fn))


def create_ply_dset( dataset_name, xyz, ply_fn, label=None, cut_threshold=[1,1,1], extra='' ):
    if extra == 'random_same_color' and type(label)==type(None):
      label = np.ones(xyz.shape[0:-1]) * np.random.randint(
        len(DatasetsMeta.g_label2class[dataset_name]))
      label = label.astype(np.int32)
    create_ply( xyz, ply_fn, label = label, label2color = DatasetsMeta.g_label2color[dataset_name], cut_threshold=cut_threshold )

def draw_points_and_edges(ply_fn, xyz, edge_indices):
  '''
  xyz: (num_point, 3)
  edge_indices:(num_edge,2)
  '''
  #
  vertex = np.zeros( shape=(xyz.shape[0]) ).astype([('x', 'f8'), ('y', 'f8'),('z', 'f8')])
  for i in range(xyz.shape[0]):
      vertex[i] = ( xyz[i,0],xyz[i,1],xyz[i,2] )

  #
  num_edge = edge_indices.shape[0]
  edge_val = np.zeros(shape=(num_edge, 5))
  edge_val[:,0:2] = edge_indices
  edge_val[:,2:5] = np.array([0,0,255])

  edge = np.zeros( shape=(edge_val.shape[0]) ).astype(
                  dtype=[('vertex1', 'i4'), ('vertex2','i4'),
                          ('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
  for i in range(edge_val.shape[0]):
      edge[i] = ( edge_val[i,0], edge_val[i,1], edge_val[i,2], edge_val[i,3], edge_val[i,4] )

  el_vertex = PlyElement.describe(vertex,'vertex')
  el_edge = PlyElement.describe(edge,'edge')
  PlyData([el_vertex, el_edge],text=True).write(ply_fn)
  print('write %s ok'%(ply_fn))

def draw_points_and_voxel_indices(ply_fn, xyz, voxel_indices):
  '''
  xyz:(num_point,3)
  voxel_indices:(num_point,3)
  '''
  # create edges
  num_point = xyz.shape[0]
  voxel_w = voxel_indices.max() + 1
  edge_indices = [ ]
  for i in range(num_point):
    for j in range(num_point):
      x_neib = (voxel_indices[j] == voxel_indices[i] + np.array([1,0,0])).all()
      y_neib = (voxel_indices[j] == voxel_indices[i] + np.array([0,1,0])).all()
      z_neib = (voxel_indices[j] == voxel_indices[i] + np.array([0,0,1])).all()
      if x_neib or y_neib or z_neib:
        edge_indices.append(np.array([[i,j]]))
  edge_indices = np.concatenate(edge_indices, 0)

  draw_points_and_edges(ply_fn, xyz, edge_indices)

def draw_blocks_by_bot_cen_top(ply_fn, bot_cen_top, random_crop=0):
  '''
    bot_cen_top: (num_blocks, 9)
  '''
  shape0 = bot_cen_top.shape
  assert shape0[-1] == 9
  bot_cen_top = np.reshape(bot_cen_top, [-1,9])
  if random_crop>0:
    N = bot_cen_top.shape[0]
    choices = np.random.choice(N, N-int(N*random_crop))
    bot_cen_top = np.take(bot_cen_top, choices, axis=0)
  block_bottom = bot_cen_top[:,0:3]
  block_center = bot_cen_top[:,3:6]
  block_top = 2*(block_center - block_bottom) + block_bottom
  box_vertexes = gen_box_8vertexs(block_bottom, block_top)
  gen_box_and_pl(ply_fn, box_vertexes, extra='random_color_between_boxes')


import  color_dic
def gen_mesh_ply(ply_fn, vertices0, vidx_per_face, face_label=None,
                 vertex_label=None, vertex_color=None,
                 extra='label_color_default'):
    '''
    '''
    assert (int(face_label is None) + int(vertex_label is None) + int(vertex_color is None)) >=2, \
      "choose one color method from three: {}, {}, {}".format(face_label, vertex_label, vertex_color)
    assert vertices0.shape[-1] ==  3
    vnpf = vidx_per_face.shape[-1]
    assert np.min(vidx_per_face)>=0, "negative vidx_per_face"

    vertices0 = np.reshape( vertices0, (-1,3) )
    vidx_per_face = np.reshape(vidx_per_face, (-1,vnpf))
    if vertex_label is not None:
      vertex_label = np.reshape(vertex_label, (-1))
    if vertex_color is not None:
      vertex_color = np.reshape(vertex_color, (-1, 3))

    num_vertex = vertices0.shape[0]
    assert np.max(vidx_per_face) < num_vertex

    is_vertex_color = (vertex_label is not None) or (vertex_color is not None)
    if not is_vertex_color:
      vertex = np.zeros( shape=(num_vertex) ).astype([('x', 'f8'), ('y', 'f8'),('z', 'f8')])
    else:
      if vertex_label is not None:
        vertex_color = np.take(color_dic.rgb_order, vertex_label, 0)

      assert vertex_color.shape[0] == num_vertex
      vertex = np.zeros( shape=(num_vertex) ).astype([('x', 'f8'), ('y', 'f8'),('z', 'f8'),
                                            ('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    for i in range(vertices0.shape[0]):
      if not is_vertex_color:
        vertex[i] = ( vertices0[i,0],vertices0[i,1],vertices0[i,2] )
      else:
        vertex[i] = ( vertices0[i,0],vertices0[i,1],vertices0[i,2], \
                      vertex_color[i,0], vertex_color[i,1], vertex_color[i,2])

    el_vertex = PlyElement.describe(vertex,'vertex')

    # define the order of the 8 vertexs for a box
    num_face = vidx_per_face.shape[0]
    is_face_color = face_label is not None
    if is_face_color:
      if face_label is not None:
        if extra=='label_color_default':
          face_label = np.squeeze(face_label)
          face_color = np.take(color_dic.rgb_order, face_label, 0)

      face = np.zeros( shape=(num_face) ).astype(
                    dtype=[('vertex_indices', 'i4', (3,)),
                           ('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    else:
      face = np.zeros( shape=(num_face) ).astype(
                    dtype=[('vertex_indices', 'i4', (vnpf,))])

    for i in range(num_face):
      if is_face_color:
        face[i] = ( vidx_per_face[i], face_color[i,0], face_color[i,1], face_color[i,2] )
      else:
        face[i] = ( vidx_per_face[i], )
    el_face = PlyElement.describe(face,'face')

    dirname = os.path.dirname(ply_fn)
    if not os.path.exists(dirname):
      os.makedirs(dirname)
    PlyData([el_vertex, el_face],text=True).write(ply_fn)
    print('write %s ok'%(ply_fn))


if __name__ == '__main__':
    test_box()

    #sg_bidxmap_i0 = np.arange( sg_bidxmap_i1.shape[0] ).reshape([-1,1,1,1])
    #sg_bidxmap_i0 = np.tile( sg_bidxmap_i0, [0,sg_bidxmap_i1.shape[1], sg_bidxmap_i1.shape[2],1] )


