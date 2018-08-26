# May 2018 xyz

from __future__ import print_function
import os
import sys, glob
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)


DATASETS = ['MATTERPORT', 'SCANNET', 'ETH', 'MODELNET40']
for ds in DATASETS:
    sys.path.append('%s/%s_util'%(ROOT_DIR,ds))

from MATTERPORT_util import MATTERPORT_Meta
from SCANNET_util import SCANNET_Meta
from ETH_util import ETH_Meta
from MODELNET_util import MODELNET40_Meta
DATASETS_Meta = [MATTERPORT_Meta, SCANNET_Meta, ETH_Meta, MODELNET40_Meta]


class DatasetsMeta():
    g_label2class = {}
    g_label_names = {}
    g_unlabelled_categories = {}
    g_easy_categories = {}
    g_label2color = {}

    for i in range(len(DATASETS)):
        DS_i = DATASETS[i]
        DS_Meta_i = DATASETS_Meta[i]
        g_label2class[DS_i] = DS_Meta_i['label2class']
        g_label_names[DS_i] = DS_Meta_i['label_names']
        g_label2color[DS_i] = DS_Meta_i['label2color']
        g_easy_categories[DS_i] = DS_Meta_i['easy_categories']
        g_unlabelled_categories[DS_i] = DS_Meta_i['unlabelled_categories']
    ##---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    g_label2class['STANFORD_INDOOR3D'] = \
                    {0:'ceiling', 1:'floor', 2:'wall', 3:'beam', 4:'column', 5:'window', 6:'door', 7:'table',
                     8:'chair', 9:'sofa', 10:'bookcase', 11:'board', 12:'clutter'}
    g_unlabelled_categories['STANFORD_INDOOR3D'] = [12]
    g_easy_categories['STANFORD_INDOOR3D'] = []
    g_label2color['STANFORD_INDOOR3D'] = \
                    {0:	[0,0,0],1:	[0,0,255],2:	[0,255,255],3: [255,255,0],4: [255,0,255],10: [100,100,255],
                    6: [0,255,0],7: [170,120,200],8: [255,0,0],9: [200,100,100],5:[10,200,100],11:[200,200,200],12:[200,200,100]}

    #---------------------------------------------------------------------------
    g_label2class['KITTI'] = {0:'background', 1:'car', 2:'pedestrian', 3:'cyclist'}   ## benz_m
    g_unlabelled_categories['KITTI'] = []
    g_label2color['KITTI'] = { 0:[0,0,0], 1:[0,0,255], 2:[0,255,255], 3:[255,255,0] }     ## benz_m
    g_easy_categories['KITTI'] = []

    def __init__(self,datasource_name):
        self.datasource_name = datasource_name
        self.label2class = self.g_label2class[self.datasource_name]
        self.label_names = [self.label2class[l] for l in range(len(self.label2class))]
        self.label2color = self.g_label2color[self.datasource_name]
        self.class2label = {cls:label for label,cls in self.label2class.iteritems()}
        self.class2color = {}
        for i in self.label2class:
            cls = self.label2class[i]
            self.class2color[cls] = self.label2color[i]
        self.num_classes = len(self.label2class)
        #self.num_classes = len(self.g_label2class) - len(self.g_unlabelled_categories[self.datasource_name])

    def get_train_test_file_list(self, data_dir, is_training):
      if self.datasource_name == "MODELNET40":
        return self.get_train_test_file_list_MODELNET(data_dir, is_training)
      if self.datasource_name == "MATTERPORT":
        return self.get_train_test_file_list_MATTERPORT(data_dir, is_training)

    def get_train_test_file_list_MATTERPORT(self, data_dir, is_training):
      from MATTERPORT_util import benchmark
      tte_scene_names = benchmark()
      split = 'train' if is_training else 'test'
      scene_names = tte_scene_names[split]
      tte_fnum = {}
      tte_fnum['train'] = 1554
      tte_fnum['test'] = 406
      tte_fnum['val'] = 234

      all_fns = glob.glob(os.path.join(data_dir, '*.tfrecord'))
      #assert len(all_fns) == 2194, len(all_fns)
      def sence_name(fn):
        return os.path.basename(fn).split('_')[0]
      the_fns = [e for e in all_fns if sence_name(e) in scene_names]
      #assert len(the_fns) == tte_fnum[split]
      return  the_fns

    def get_train_test_file_list_MODELNET(self, data_dir, is_training):
      from MODELNET_util import train_names, test_names
      if is_training:
        train_names = train_names()
        train_fns = [os.path.join(data_dir, e+'.tfrecord') for e in train_names]
        # Check exist
        for e in train_fns[0:len(train_fns):100]:
          assert( os.path.exists(e) )
        assert len(train_fns) == 9843

        return train_fns

      else:
        test_names = test_names()
        test_fns = [os.path.join(data_dir, e+'.tfrecord') for e in test_names]
        for e in test_fns[0:len(test_fns):10]:
          assert( os.path.exists(e) )
        assert len(test_fns) == 2468

        return test_fns




def show_all_colors( datasource_name ):
    from PIL import Image
    dset_meta = DatasetsMeta(datasource_name)
    label2color = dset_meta.label2color
    label2class = dset_meta.label2class
    path = os.path.join( BASE_DIR,'label_colors_'+datasource_name )
    if not os.path.exists(path):
        os.makedirs(path)
    for label,color in label2color.iteritems():
        if label < len( label2class ):
            cls = label2class[label]
        else:
            cls = 'empty'
        data = np.zeros((512,512,3),dtype=np.uint8)
        color_ = np.array(color,dtype=np.uint8)
        data += color_
        img = Image.fromarray(data,'RGB')
        fn = path+'/'+str(label)+'_'+cls+'.png'
        img.save(fn)
        print(fn)
        img.show()

if __name__ == '__main__':
  show_all_colors('MATTERPORT')

