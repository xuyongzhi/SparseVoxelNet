#!/bin/bash

gpu_id=0
num_gpus=1
batch_size=32

#learning_rate0=0.001
#optimizer='adam'
#lr_decay_epochs=20

learning_rate0=0.01
optimizer='momentum'
lr_decay_epochs=15

lr_decay_rate=0.7
batch_norm_decay0=0.5
feed_data='xyzs'
drop_imo='0_0_5'
aug_types='N'
use_bias=1
block_style='PointNet'
shortcut='MZ'
loss_lw_gama=-1
train_epochs=101
model_flag='m'
residual=0
resnet_size=9



#./modelnet.sh   $aug_types  $batch_size $model_flag $learning_rate0 $num_gpus $feed_data $drop_imo  $optimizer $use_bias $lr_decay_epochs $lr_decay_rate $resnet_size $block_style $residual $shortcut $loss_lw_gama  $train_epochs  $batch_norm_decay0 $gpu_id
#
#
#lr_decay_epochs=20
#./modelnet.sh   $aug_types  $batch_size $model_flag $learning_rate0 $num_gpus $feed_data $drop_imo  $optimizer $use_bias $lr_decay_epochs $lr_decay_rate $resnet_size $block_style $residual $shortcut $loss_lw_gama  $train_epochs  $batch_norm_decay0 $gpu_id
#lr_decay_epochs=10



gpu_id=1

learning_rate0=0.001
optimizer='adam'
lr_decay_epochs=20
./modelnet.sh   $aug_types  $batch_size $model_flag $learning_rate0 $num_gpus $feed_data $drop_imo  $optimizer $use_bias $lr_decay_epochs $lr_decay_rate $resnet_size $block_style $residual $shortcut $loss_lw_gama  $train_epochs  $batch_norm_decay0 $gpu_id

learning_rate0=0.001
optimizer='momentum'
lr_decay_epochs=10
#./modelnet.sh   $aug_types  $batch_size $model_flag $learning_rate0 $num_gpus $feed_data $drop_imo  $optimizer $use_bias $lr_decay_epochs $lr_decay_rate $resnet_size $block_style $residual $shortcut $loss_lw_gama  $train_epochs  $batch_norm_decay0 $gpu_id
learning_rate0=0.01


#----------------------------------------
#aug_types='r-360_0_0'
#aug_types='rpsfj-360_0_0'
#aug_types='psfj'
#aug_types='l'
#optimizer='adam'
#feed_data='xyzrsg-nxnynz'
#----------------------------------------



