import os,csv
import numpy as np
import zipfile,gzip
from plyfile import PlyData, PlyElement
import glob
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
mpcat40_fn = BASE_DIR+'/mpcat40.tsv'
category_mapping_fn = BASE_DIR+'/category_mapping.tsv'

def hexcolor2int(hexcolor):
    hexcolor = hexcolor[1:]
    hex0 = '0x'+hexcolor[0:2]
    hex1 = '0x'+hexcolor[2:4]
    hex2 = '0x'+hexcolor[4:6]
    hex_int = [int(hex0,16),int(hex1,16),int(hex2,16)]
    return hex_int

MATTERPORT_Meta = {}
MATTERPORT_Meta['label2class'] = {}
MATTERPORT_Meta['label2color'] = {}
MATTERPORT_Meta['label2WordNet_synset_key'] = {}
MATTERPORT_Meta['label2NYUv2_40label'] = {}
MATTERPORT_Meta['label25'] = {}
MATTERPORT_Meta['label26'] = {}

MATTERPORT_Meta['unlabelled_categories'] = [0,41]
MATTERPORT_Meta['easy_categories'] = []

bad_files = [('YFuZgdQ5vWj','region19'), ('VFuaQ6m2Qom','region38'), ('VFuaQ6m2Qom','region40')] # all void
MATTERPORT_Meta['bad_files'] = ['%s/%s/region_segmentations/%s.ply'%(e[0], e[0], e[1]) for e in bad_files]

with open(mpcat40_fn,'r') as f:
    reader = csv.reader(f,delimiter='\t')
    for i,line in enumerate(reader):
        if i==0: continue
        k = int(line[0])
        MATTERPORT_Meta['label2class'][k] = line[1]
        MATTERPORT_Meta['label2color'][k] = hexcolor2int( line[2] )
        MATTERPORT_Meta['label2WordNet_synset_key'][k]=line[3]
        MATTERPORT_Meta['label2NYUv2_40label'][k] = line[4]
        MATTERPORT_Meta['label25'][k] = line[5]
        MATTERPORT_Meta['label26'][k] = line[6]

MATTERPORT_Meta['label_names'] = [MATTERPORT_Meta['label2class'][l] for l in range(len(MATTERPORT_Meta['label2class'])) ]

Material_Index = 0
Instance_Index = 1
Category_Index = 2

with open(category_mapping_fn,'r') as f:
    '''
    ['index', 'raw_category', 'category', 'count', 'nyuId', 'nyu40id', 'eigen13id', 'nyuClass', 'nyu40class', 'eigen13class',
    'ModelNet40', 'ModelNet10', 'ShapeNetCore55', 'synsetoffset', 'wnsynsetid', 'wnsynsetkey', 'mpcat40index', 'mpcat40']
    '''
    reader = csv.reader(f,delimiter='\t')

    mapping_vals = {}
    for i,line in enumerate(reader):
        if i==0:
            ele_names = line
            #print(ele_names)
            #print('\n')
            for j in range(18):
                mapping_vals[ele_names[j]] = []
        else:
            for j in range(18):
                mapping_vals[ele_names[j]].append( line[j] )
                #print(ele_names[j])
                #print(mapping_vals[ele_names[j]])
    mapping_vals['index'] = [int(v) for v in mapping_vals['index']]
    mapping_vals['mpcat40index'] = [int(v) for v in mapping_vals['mpcat40index']]

    assert mapping_vals['index'] == list( range(1,1+len(mapping_vals['index'])) )

    #print(mapping_vals['raw_category'])
    #print(mapping_vals['category'])
    #print(mapping_vals['mpcat40index'])
    #print(mapping_vals['mpcat40'])

rawcategory_2_mpcat40ind = mapping_vals['mpcat40index']
rawcategory_2_mpcat40 = mapping_vals['mpcat40']

def get_cat40_from_rawcat(raw_category_indexs):
    '''
    raw_category_indexs.shape=[num_point]
    '''
    assert raw_category_indexs.ndim==1
    mpcat40_idxs = np.zeros(shape=raw_category_indexs.shape, dtype=np.int32)
    num_point = raw_category_indexs.shape[0]
    mpcat40s =['']*num_point
    for j in range(num_point):
        raw_category_index = int(raw_category_indexs[j])

        if raw_category_index<=0:
            mpcat40_j = 'void'
            mpcat40_idx_j = 0
        else:
            assert raw_category_index>0, "raw_category_index start from 1"
            mpcat40_j = rawcategory_2_mpcat40[raw_category_index-1]
            mpcat40_idx_j = rawcategory_2_mpcat40ind[raw_category_index-1]
        mpcat40_idxs[j] = mpcat40_idx_j
        assert mpcat40_j == MATTERPORT_Meta['label2class'][mpcat40_idx_j],"%s != %s"%(mpcat40_j,MATTERPORT_Meta['label2class'][mpcat40_idx_j])
        mpcat40s[j] += mpcat40_j
    return mpcat40_idxs

def benchmark():
  tte_names = {}
  tte_names['train'] = np.loadtxt( os.path.join(BASE_DIR,'scenes_train.txt'), dtype=str )
  tte_names['test'] = np.loadtxt( os.path.join(BASE_DIR,'scenes_test.txt'), dtype=str )
  tte_names['val'] = np.loadtxt( os.path.join(BASE_DIR,'scenes_val.txt'), dtype=str )
  #for item in tte_names:
  #  print('{} {}'.format(item, len(tte_names[item])))
  return tte_names


def parse_ply_file(ply_fn):
  '''
  element vertex 1522546
  property float x
  property float y
  property float z
  property float nx
  property float ny
  property float nz
  property float tx
  property float ty
  property uchar red
  property uchar green
  property uchar blue

  element face 3016249
  property list uchar int vertex_indices
  property int material_id
  property int segment_id
  property int category_id
  '''
  with open(ply_fn, 'r') as ply_fo:
    plydata = PlyData.read(ply_fo)
    num_ele = len(plydata.elements)
    num_vertex = plydata['vertex'].count
    num_face = plydata['face'].count
    data_vertex = plydata['vertex'].data
    data_face = plydata['face'].data

    ## vertex
    vertex_eles = ['x','y','z','nx','ny','nz','tx','ty','red','green','blue']
    datas_vertex = {}
    for e in vertex_eles:
        datas_vertex[e] = np.expand_dims(data_vertex[e],axis=-1)
    vertex_xyz = np.concatenate([datas_vertex['x'],datas_vertex['y'],datas_vertex['z']],axis=1)
    vertex_nxnynz = np.concatenate([datas_vertex['nx'],datas_vertex['ny'],datas_vertex['nz']],axis=1)
    vertex_rgb = np.concatenate([datas_vertex['red'],datas_vertex['green'],datas_vertex['blue']],axis=1)

    ## face
    vertex_idx_per_face = data_face['vertex_indices']
    vertex_idx_per_face = np.concatenate(vertex_idx_per_face,axis=0)
    vertex_idx_per_face = np.reshape(vertex_idx_per_face,[-1,3])

    datas = {}
    datas['xyz'] = vertex_xyz           # (N,3)
    datas['nxnynz'] = vertex_nxnynz     # (N,3)
    datas['color'] = vertex_rgb         # (N,3)
    datas['vidx_per_face'] = vertex_idx_per_face  # (F,3)
    datas['label_material'] = np.expand_dims(data_face['material_id'],1) # (F,1)
    datas['label_instance'] = np.expand_dims(data_face['segment_id'],1)
    datas['label_raw_category'] = np.expand_dims(data_face['category_id'],1)
    label_category = get_cat40_from_rawcat(data_face['category_id'])
    datas['label_category'] = np.expand_dims(label_category, 1)

    return datas

    face_eles = ['vertex_indices','material_id','segment_id','category_id']
    datas_face = {}
    for e in face_eles:
        datas_face[e] = np.expand_dims(data_face[e],axis=-1)
    face_semantic = np.concatenate([datas_face['material_id'], datas_face['segment_id'], datas_face['category_id']],axis=1)

    return vertex_xyz, vertex_nxnynz, vertex_rgb, vertex_idx_per_face, face_semantic

def parse_ply_vertex_semantic(ply_fn):
    vertex_xyz, vertex_nxnynz, vertex_rgb, vertex_idx_per_face, face_semantic = parse_ply_file(ply_fn)

    num_vertex = vertex_xyz.shape[0]
    vertex_semantic,vertex_indices_multi_semantic,face_indices_multi_semantic =\
      get_vertex_label_from_face(vertex_idx_per_face,face_semantic,num_vertex)

    IsDeleteNonPosId = False
    if IsDeleteNonPosId:
      # get vertex and face  with category_id==-1 or ==0
      vertex_nonpos_indices = np.where(vertex_semantic[:, Category_Index]<1)[0]
      face_nonpos_indices = np.where(face_semantic[:,Category_Index]<1)[0]

      vertex_del_indices = np.concatenate([vertex_indices_multi_semantic, vertex_nonpos_indices], 0)
      face_del_indices = np.concatenate([face_indices_multi_semantic, face_nonpos_indices], 0)
    else:
      vertex_del_indices = vertex_indices_multi_semantic
      face_del_indices = face_indices_multi_semantic

    # Del VexMultiSem
    vertex_xyz = np.delete(vertex_xyz, vertex_del_indices, axis=0)
    vertex_nxnynz = np.delete(vertex_nxnynz, vertex_del_indices, axis=0)
    vertex_rgb = np.delete(vertex_rgb, vertex_del_indices, axis=0)
    vertex_semantic = np.delete(vertex_semantic, vertex_del_indices,axis=0)
    vertex_idx_per_face = np.delete(vertex_idx_per_face, face_del_indices, axis=0)
    face_semantic = np.delete(face_semantic, face_del_indices, axis=0)

    return vertex_xyz, vertex_nxnynz, vertex_rgb, vertex_semantic, vertex_idx_per_face, face_semantic

def get_vertex_label_from_face(vertex_idx_per_face, face_semantic, num_vertex):
    '''
    vertex_idx_per_face: the vertex indices in each face
    vertex_face_indices: the face indices in each vertex
    '''
    vertex_face_indices = -np.ones(shape=[num_vertex,30])
    face_num_per_vertex = np.zeros(shape=[num_vertex]).astype(np.int8)
    vertex_semantic = np.zeros(shape=[num_vertex,3]) # only record the first one
    vertex_semantic_num = np.zeros(shape=[num_vertex])
    vertex_indices_multi_semantic = set()
    face_indices_multi_semantic = set()
    for i in range(vertex_idx_per_face.shape[0]):
        for vertex_index in vertex_idx_per_face[i]:
            face_num_per_vertex[vertex_index] += 1
            vertex_face_indices[vertex_index,face_num_per_vertex[vertex_index]-1] = i

            if vertex_semantic_num[vertex_index] == 0:
                vertex_semantic_num[vertex_index] += 1
                vertex_semantic[vertex_index] = face_semantic[i]
            else:
                # (1) Only 60% vertexs have unique labels for all three semntics
                # (2) There are 96% vertexs have unique labels for the first two:  category_id and segment_id
                # (3) only need category_id to be same
                IsSameSemantic = (vertex_semantic[vertex_index][Category_Index]==face_semantic[i][Category_Index]).all()
                if not IsSameSemantic:
                    vertex_semantic_num[vertex_index] += 1
                    vertex_indices_multi_semantic.add(vertex_index)
                    face_indices_multi_semantic.add(i)
    vertex_indices_multi_semantic = np.array(list(vertex_indices_multi_semantic))
    face_indices_multi_semantic = np.array(list(face_indices_multi_semantic))
    print('vertex rate with multiple semantic: %f'%(1.0*vertex_indices_multi_semantic.shape[0]/num_vertex))

    vertex_semantic = vertex_semantic.astype(np.int32)

   # vertex_semantic_num_max = np.max(vertex_semantic_num)
   # vertex_semantic_num_min = np.min(vertex_semantic_num)
   # vertex_semantic_num_mean = np.mean(vertex_semantic_num)
   # vertex_semantic_num_one = np.sum(vertex_semantic_num==1)
   # print(vertex_semantic_num_max)
   # print(vertex_semantic_num_mean)
   # print(vertex_semantic_num_min)
   # print(1.0*vertex_semantic_num_one/num_vertex)

    return vertex_semantic,vertex_indices_multi_semantic,face_indices_multi_semantic


def zip_extract(ply_item_name,zipf,house_dir_extracted):
    '''
    extract file if not already
    '''
    #zipfile_name = '%s/%s/%s.%s'%(house_name,groupe_name,file_name,file_format)
    file_path = house_dir_extracted + '/' + ply_item_name
    noneed_extract = False
    if os.path.exists(file_path):
      try:
        with open(file_path, 'r') as ply_fo:
          plydata = PlyData.read(ply_fo)
        noneed_extract = True
      except:
        print('\nfile extracted but not intact: %s\n'%(file_path))

    if not noneed_extract:
        print('extracting %s...'%(file_path))
        file_path_extracted  = zipf.extract(ply_item_name,house_dir_extracted)
        print('file extracting finished: %s'%(file_path_extracted) )
        assert file_path == file_path_extracted
    else:
        print('file already extracted: %s'%(file_path))
    return file_path

def unzip_house(house_name):
  house_dir = self.scans_dir+'/%s'%(house_name)
  house_dir_extracted = self.matterport3D_extracted_dir + self.scans_name+'/%s'%(house_name)
  region_segmentations_zip_fn = house_dir+'/region_segmentations.zip'
  rs_zf = zipfile.ZipFile(region_segmentations_zip_fn,'r')

  namelist_ply = [ name for name in rs_zf.namelist()  if 'ply' in name]
  num_region = len(namelist_ply)

  for ply_item_name in namelist_ply:
    results = pool.apply_async(WriteRawH5f_Region_Ply,(ply_item_name,rs_zf, house_name, self.scans_h5f_dir, house_dir_extracted))
    s = ply_item_name.index('region_segmentations/region')+len('region_segmentations/region')
    e = ply_item_name.index('.ply')
    k_region = int( ply_item_name[ s:e ] )
    print('apply_async %d'%(k_region))

def unzip_all():
  house_names_ls = os.listdir('/DS/Matterport3D/Matterport3D_WHOLE/v1/scans')
  for house_name in house_names_ls:
    unzip_house(house_name)

if __name__ == '__main__':
  unzip_all()
