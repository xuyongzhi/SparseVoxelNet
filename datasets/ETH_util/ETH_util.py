# May 2018 xyz

import os
import numpy as np
import glob
import multiprocessing as mp


ETH_Meta = {}

ETH_Meta['label2class'] = {0: 'unlabeled points', 1: 'man-made terrain', 2: 'natural terrain',\
                    3: 'high vegetation', 4: 'low vegetation', 5: 'buildings', \
                    6: 'hard scape', 7: 'scanning artefacts', 8: 'cars'}
ETH_Meta['unlabelled_categories'] = [0]
ETH_Meta['easy_categories'] = []
ETH_Meta['label2color'] = \
                {0:	[0,0,0],1:	[0,0,255],2:	[0,255,255],3: [255,255,0],4: [255,0,255],
                6: [0,255,0],7: [170,120,200],8: [255,0,0],5:[10,200,100]}
ETH_Meta['label_names'] = [ETH_Meta['label2class'][l] for l in range(len(ETH_Meta['label2class']))]


def parse_raw_ETH( fn_txt ):
    base_fn = os.path.splitext( fn_txt )[0]
    fn_labels = base_fn+'.labels'

    # {x, y, z, intensity, r, g, b}
    raw_data = np.loadtxt( fn_txt )
    #xyz = raw_data[:,0:3]
    #intensity = raw_data[:,3:4]
    #rgb = raw_data[:,4:7]
    if os.path.exists( fn_labels ):
        labels = np.loadtxt(fn_labels).reshape( (-1,1) )
        assert raw_data.shape[0] == labels.shape[0]
    else:
        #labels = np.ones(shape=(xyz.shape[0],1))*(-111)
        labels = None
    return raw_data[:,0:3], raw_data[:,3:4], raw_data[:,4:7], labels

def ExtractAll():
    ETH_DIR = '../ETH'
    fn_7z_ls = glob.glob( ETH_DIR + '/*.7z' )
    print(len(fn_7z_ls))

    for fn_7z in fn_7z_ls:
        base_fn = os.path.splitext( fn_7z )[0]
        if base_fn[-3:] == 'txt':
            base_fn = os.path.splitext(base_fn)[0]
        basename = os.path.basename( base_fn )
        if basename == 'sem8_labels_training':
            # labels 7z, extract manually
            continue
        fn_txt = base_fn+'.txt'
        if not os.path.exists( fn_txt ):
            print('not exist: %s'%(fn_txt))
            IsExtract = True
        else:
            txt_size = os.path.getsize( fn_txt )
            size_7z = os.path.getsize( fn_7z )
            compress_rate =1.0 * txt_size / size_7z
            print('compress_rate:%f'%(compress_rate))
            min_compress_rate = 4.5
            if basename == 'castleblatten_station5_xyz_intensity_rgb':
                min_compress_rate = 3.6
            if compress_rate < min_compress_rate:
                print('not intact: %s'%(fn_txt))
                os.system( 'rm %s'%(fn_txt) )
                IsExtract = True
            else:
                print('txt intact: %s'%(fn_txt))
                IsExtract = False
        if  IsExtract:
            os.system( '7za e %s -o%s'%(fn_7z, os.path.dirname(fn_7z)) )
            if basename == 'neugasse_station1_xyz_intensity_rgb':
                # extracted txt name is incoorect
                incorrect_txt_fn = ETH_DIR + '/station1_xyz_intensity_rgb.txt'
                os.system( 'mv %s %s'%( incorrect_txt_fn, fn_txt ) )



if __name__=='__main__':
    ExtractAll()
