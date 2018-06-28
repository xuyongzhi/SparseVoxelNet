import os,csv
import numpy as np

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
    mpcat40_idxs = np.zeros(shape=raw_category_indexs.shape)
    num_point = raw_category_indexs.shape[0]
    mpcat40s =['']*num_point
    for j in range(num_point):
        raw_category_index = int(raw_category_indexs[j])

        if raw_category_index==0:
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

