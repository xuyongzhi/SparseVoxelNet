#  created by Xuesong li, on 12/5/2018

import numpy as np
import os
import sys

ROOT_DIR =  os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR,'utils_benz'))
sys.path.append(os.path.join(ROOT_DIR,'config'))

from utils_voxelnet import label_to_gt_box3d

Max_bounding_box_num = 60
Bounding_box_channel = 7


def extract_bounding_box(pl_sph5_filename, g_xyz_center, g_xyz_bottom, g_xyz_top, file_datas, file_gbixyzs, file_rootb_split_idxmaps ):
    '''
    two function:
        1, keeping the sph5 file which contains the bounding boxes;
        2, delete the block without any bounding box
    step1
        1, reading the g_xyz_bottom and g_xyz_top
        2, check whether there is bounding box inside the block,
        3, if not check the next block
        4, if yes, check whether other bounding box also resides in this box
        5, save the file_data into new_file_data
        6, save the all bounding box into file_boudning_box, and the first row is the num of bounding box within the block
    '''
    file_path = os.path.dirname(os.path.dirname(os.path.dirname(pl_sph5_filename)))
    label_name = os.path.basename(pl_sph5_filename).replace(".sph5",".txt")
    label_file_path = file_path + '/label/' + label_name
    assert os.path.exists(label_file_path)
    label_data = reading_label_data(label_file_path)  ## reading all the label_data
    num_label = label_data.shape[0]

    num_blocks = file_datas.shape[0]
    #num_points = file_datas.shape[1]
    #num_point_channel = file_datas.shape[2]

    file_bounding_boxes = np.empty((0, Max_bounding_box_num, Bounding_box_channel), dtype=np.float32)

    del_gb_ls = []
    for block_id in range(num_blocks):
        x_min = g_xyz_bottom[block_id][0]
        y_min = g_xyz_bottom[block_id][1]
        z_min = g_xyz_bottom[block_id][2]
        x_max = g_xyz_top[block_id][0]
        y_max = g_xyz_top[block_id][1]
        z_max = g_xyz_top[block_id][2]
        temp_bounding_boxes = np.zeros((1, Max_bounding_box_num, Bounding_box_channel), dtype=np.float32)  ## inilization every time

        counter_num_label = 0
        for label_id in range(num_label):
            if label_data[label_id, 0] > x_min  and label_data[label_id, 0] < x_max:  ## check x scope
                if label_data[label_id, 1] > y_min and label_data[label_id, 1] < y_max:  ## check y scope
                    if label_data[label_id, 2] > z_min and label_data[label_id, 2] < z_max:  ## check z scope
                        counter_num_label = counter_num_label + 1
                        temp_bounding_boxes[0,counter_num_label,...] = label_data[label_id,...]

        if counter_num_label > 0:
            temp_bounding_boxes[0,0,0] = counter_num_label
            file_bounding_boxes = np.append(file_bounding_boxes, temp_bounding_boxes, axis=0)

        else:
            del_gb_ls.append( block_id )
    file_datas = np.delete( file_datas, del_gb_ls, 0 )
    file_gbixyzs = np.delete( file_gbixyzs, del_gb_ls, 0 )
    file_rootb_split_idxmaps = np.delete( file_rootb_split_idxmaps, del_gb_ls, 0 )

    return file_bounding_boxes, file_datas, file_gbixyzs, file_rootb_split_idxmaps



def reading_label_data(label_file_path):

    label = [np.array([line for line in open( label_file_path, 'r').readlines()])]  # (N')].i
    label_data_i = label_to_gt_box3d( label, cls='Car', coordinate = 'lidar')

    return label_data_i[0]



def rm_gb_with_no_boundingbox( pl_sph5_filename, all_sorted_larger_aimbids, gb_center, gb_bottom, gb_top):
    file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(pl_sph5_filename))))
    label_name = os.path.basename(pl_sph5_filename).replace(".bmh5",".txt")
    label_file_path = file_path + '/label/' + label_name
    assert os.path.exists(label_file_path)
    label_data = reading_label_data(label_file_path)  ## reading all the label_data
    num_label = label_data.shape[0]

    num_blocks = gb_bottom.shape[0]

    select_indexes = []
    for block_id in range(num_blocks):
        x_min = gb_bottom[block_id][0]
        y_min = gb_bottom[block_id][1]
        z_min = gb_bottom[block_id][2]
        x_max = gb_top[block_id][0]
        y_max = gb_top[block_id][1]
        z_max = gb_top[block_id][2]


        for label_id in range(num_label):
            if label_data[label_id, 0] > x_min  and label_data[label_id, 0] < x_max:  ## check x scope
                if label_data[label_id, 1] > y_min and label_data[label_id, 1] < y_max:  ## check y scope
                    if label_data[label_id, 2] > z_min and label_data[label_id, 2] < z_max:  ## check z scope
                        select_indexes.append(all_sorted_larger_aimbids[block_id])
                        break

    return select_indexes




