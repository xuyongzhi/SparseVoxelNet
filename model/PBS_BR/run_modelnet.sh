batch_size=64
model_flag='V'
learning_rate0=0.001
lr_decay_epochs=10
lr_decay_rate=0.7
num_gpus=2
feed_data='xyzs-nxnynz'
drop_imo='0_0_5'
optimizer='momentum'
aug_types='N'
use_bias=1

#--------------------------------
resnet_size=37
block_style='Bottleneck'
batch_size=48
#--------------------------------
resnet_size=36
block_style='Bottleneck'
batch_size=64
#--------------------------------
resnet_size=36
block_style='Regular'
batch_size=64
#--------------------------------
resnet_size=37
block_style='Regular'
batch_size=32

./modelnet.sh   $aug_types  $batch_size $model_flag $learning_rate0 $num_gpus $feed_data $drop_imo  $optimizer $use_bias $lr_decay_epochs $lr_decay_rate $resnet_size $block_style



#----------------------------------------
#aug_types='rpsfj-360_0_0'
#aug_types='psfj'
#optimizer='adam'
#feed_data='xyzrsg-nxnynz'
#----------------------------------------



