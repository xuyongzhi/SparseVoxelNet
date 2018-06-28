from __future__ import print_function
import glob,os,sys
import zipfile,gzip
from plyfile import PlyData, PlyElement

MATTERPORT3D_ROOTDIR = '/home/y/DS/Matterport3D/Matterport3D_Whole'
SCANS_DIR = MATTERPORT3D_ROOTDIR+'/v1/scans'



def read_house(house_name):
    #readzip_house_segmentation(SCANS_DIR,house_name)
    read_region_segmentations(SCANS_DIR,house_name)

def read_region_segmentations(scans_dir,house_name):
    house_dir = scans_dir+'/%s'%(house_name)
    region_segmentations_zip_fn = house_dir+'/region_segmentations.zip'
    rs_zf = zipfile.ZipFile(region_segmentations_zip_fn,'r')
    num_region = len(rs_zf.namelist())/4
    #print (rs_zf.namelist())
    print('region num = %d'%(num_region))

    for k in range(num_region):
        ply_path = zip_extract('region_segmentations',house_name,rs_zf,house_dir,'ply')
        with open(ply_path,'r') as ply_fo:
            parse_ply_file(ply_fo)

def readzip_house_segmentation(scans_dir,house_name):
    house_dir = scans_dir+'/%s'%(house_name)
    house_segmentation_zip_fn = house_dir+'/house_segmentations.zip'
    hs_zf = zipfile.ZipFile(house_segmentation_zip_fn,'r')
    print (hs_zf.namelist())

    with hs_zf.open('%s/house_segmentations/%s.house'%(house_name,house_name)) as house_fo:
        parse_house_file(house_fo)
    ply_path = zip_extract('house_segmentations',house_name,hs_zf,house_dir,'ply')

    with open(ply_path,'r') as ply_fo:
        parse_ply_file(ply_fo)

def zip_extract(groupname,house_name,zipf,house_dir,file_format='ply'):
    '''
    extract file if not already
    '''
    file_name = '%s/%s/%s.%s'%(house_name,groupname,house_name,file_format)
    file_path = house_dir + '/' + file_name
    if not os.path.exists(file_path):
        print('extracting %s...'%(file_name))
        file_path_extracted  = zipf.extract(file_name,house_dir)
        print('file extracting finished: %s'%(file_path_extracted) )
        assert file_path == file_path_extracted
    else:
        print('file file already extracted: %s'%(file_path))
    return file_path

def parse_ply_file(ply_fo):
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
    plydata = PlyData.read(ply_fo)
    num_ele = len(plydata.elements)
    num_vertex = plydata['vertex'].count
    data_vertex = plydata['vertex'].data
    data_face = plydata['face'].data
    print(plydata.elements[0])
    print('\n')
    print(plydata.elements[1])
    #xyz = plydata['vertex']
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    print(data_face[0:2])

def parse_house_file(house_fo):
    for i,line in enumerate( house_fo ):
        if i<1:
            print(line)
        break



if __name__ == '__main__':
    house_name = '17DRP5sb8fy'
    #view_house(house_name)
    read_house(house_name)
