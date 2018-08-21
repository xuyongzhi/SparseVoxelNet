#!/bin/bash

gpu_id=0
num_gpus=1
batch_size=32

learning_rate0=0.001
optimizer='adam'
lr_decay_epochs=10

#learning_rate0=0.01
#optimizer='momentum'
#lr_decay_epochs=15

lr_decay_rate=0.7
batch_norm_decay0=0.6
feed_data='xyzs'
use_xyz=1
drop_imo='0_0_5'
aug_types='N'
use_bias=1
block_style='Regular'
shortcut='MC'
loss_lw_gama=-1
train_epochs=61
residual=0
resnet_size='20A'

model_flag='m'



./modelnet.sh   $aug_types  $batch_size $model_flag $learning_rate0 $num_gpus $feed_data $drop_imo  $optimizer $use_bias $lr_decay_epochs $lr_decay_rate $resnet_size $block_style $residual $shortcut $loss_lw_gama  $train_epochs  $batch_norm_decay0 $gpu_id $use_xyz


#----------------------------------------
#aug_types='r-360_0_0'
#aug_types='rpsfj-360_0_0'
#aug_types='rlpsfj-360_0_0'
#aug_types='psfj'
#aug_types='l'
#optimizer='adam'
#feed_data='xyzrsg-nxnynz'
#----------------------------------------



