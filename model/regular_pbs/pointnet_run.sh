#!/bin/bash

num_gpus=2
gpu_id=1
batch_size=10
train_epochs=401

learning_rate0=0.001
optimizer='adam'
lr_decay_epochs=15
lr_decay_rate=0.7
batch_norm_decay0=0.6

#learning_rate0=0.01
#optimizer='momentum'
#lr_decay_epochs=15


feed_data='xyzs'
use_xyz=1
aug_types='N'
loss_lw_gama=-1
drop_imo='0_0_5'
use_bias=1

block_style='Regular'
shortcut='MC'
residual=0

resnet_size='1A17'
sg_flag='10240_2'
model_flag='m'


#feed_data='xyzs-nxnynz'

./base.sh   $aug_types  $batch_size $model_flag $learning_rate0 $num_gpus $feed_data $drop_imo  $optimizer $use_bias $lr_decay_epochs $lr_decay_rate $resnet_size $block_style $residual $shortcut $loss_lw_gama  $train_epochs  $batch_norm_decay0 $gpu_id $use_xyz $sg_flag



#----------------------------------------
#aug_types='r-360_0_0'
#aug_types='rpsfj-360_0_0'
#aug_types='rlpsfj-360_0_0'
#aug_types='psfj'
#aug_types='l'
#optimizer='adam'
#feed_data='xyzrsg-nxnynz'
#----------------------------------------



