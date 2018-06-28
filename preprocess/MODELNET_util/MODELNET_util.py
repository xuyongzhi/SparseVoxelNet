# May 2018 xyz

import os, sys
import numpy as np

BASE_DIR = os.path.dirname( os.path.abspath(__file__) )



shape_names = np.loadtxt( os.path.join(BASE_DIR,'modelnet40_shape_names.txt'), dtype=str )
N = len(shape_names)

MODELNET40_Meta = {}
MODELNET40_Meta['label_names'] = shape_names
MODELNET40_Meta['label2class'] = { i:shape_names[i] for i in range(N) }
MODELNET40_Meta['unlabelled_categories'] = []
MODELNET40_Meta['easy_categories'] = []

sys.path.append( os.path.join(BASE_DIR,'../MATTERPORT_util') )
from MATTERPORT_util import MATTERPORT_Meta

MODELNET40_Meta['label2color'] = MATTERPORT_Meta['label2color']
del MODELNET40_Meta['label2color'][40]
del MODELNET40_Meta['label2color'][41]



