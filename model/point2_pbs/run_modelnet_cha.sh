#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
gpu_id=0
num_gpus=2
batch_size=32
learning_rate0=0.001
optimizer='adam'
lr_decay_epochs=20
lr_decay_rate=0.7
feed_data='xyzs'
drop_imo='0_0_5'
aug_types='N'
use_bias=1
block_style='PointNet'
shortcut='MZ'
loss_lw_gama=-1
train_epochs=151
model_flag='m'
residual=0

resnet_size=9

#batch_size=64
#feed_data='xyzrsg-nxnynz'



aug_types='rpsfj-360_0_0'
./modelnet.sh   $aug_types  $batch_size $model_flag $learning_rate0 $num_gpus $feed_data $drop_imo  $optimizer $use_bias $lr_decay_epochs $lr_decay_rate $resnet_size $block_style $residual $shortcut $loss_lw_gama  $train_epochs $gpu_id
#
#
#aug_types='r-360_0_0'
#./modelnet.sh   $aug_types  $batch_size $model_flag $learning_rate0 $num_gpus $feed_data $drop_imo  $optimizer $use_bias $lr_decay_epochs $lr_decay_rate $resnet_size $block_style $residual $shortcut $loss_lw_gama  $train_epochs  $gpu_id
#
#
#aug_types='psfj'
#./modelnet.sh   $aug_types  $batch_size $model_flag $learning_rate0 $num_gpus $feed_data $drop_imo  $optimizer $use_bias $lr_decay_epochs $lr_decay_rate $resnet_size $block_style $residual $shortcut $loss_lw_gama  $train_epochs  $gpu_id
#aug_types='N'






#----------------------------------------
#aug_types='r-360_0_0'
#aug_types='rpsfj-360_0_0'
#aug_types='psfj'
#optimizer='adam'
#feed_data='xyzrsg-nxnynz'
#----------------------------------------



