#!/bin/bash
#PBS -q gpu
#PBS -l walltime=20:00:00
#PBS -l mem=16GB
#PBS -l jobfs=0GB
#PBS -l ngpus=2
#PBS -l ncpus=6
## For licensed software, you have to specify it to get the job running. For unlicensed software, you should also specify it to help us analyse the software usage on our system.
#PBS -l software=tensorflow/1.8-cudnn7.1-python2.7

## The job will be executed from current working directory instead of home.
#PBS -l wd 
#PBS -r y
##PBS -M y.xu@unsw.edu.au
##PBS -m abe

module load  tensorflow/1.8-cudnn7.1-python2.7
module list
 

#------------------------------------------------------------------------
residual=1
batch_size=32
learning_rate0=0.001
lr_decay_epochs=10
lr_decay_rate=0.7
num_gpus=2
feed_data='xyzs-nxnynz'
drop_imo='0_0_5'
optimizer='momentum'
aug_types='N'
use_bias=1
block_style='Regular'
residual=1

model_flag='V'
resnet_size=37
shortcut='C'

./modelnet.sh   $aug_types  $batch_size $model_flag $learning_rate0 $num_gpus $feed_data $drop_imo  $optimizer $use_bias $lr_decay_epochs $lr_decay_rate $resnet_size $block_style $residual $shortcut
