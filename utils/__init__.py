import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
import pdb; pdb.set_trace()  # XXX BREAKPOINT
DATASETS_DIR = os.path.join(ROOT_DIR, 'datasets')
