residual=0
batch_size=32
learning_rate0=0.01
lr_decay_epochs=10
lr_decay_rate=0.7
num_gpus=2
feed_data='xyzs'
drop_imo='0_0_5'
optimizer='adam'
aug_types='N'
use_bias=1
block_style='Regular'
shortcut='MZ'
loss_lw_gama=1
train_epochs=41
model_flag='m'
loss_lw_gama=1
resnet_size=24

aug_types='r-360_0_0'

./modelnet.sh   $aug_types  $batch_size $model_flag $learning_rate0 $num_gpus $feed_data $drop_imo  $optimizer $use_bias $lr_decay_epochs $lr_decay_rate $resnet_size $block_style $residual $shortcut $loss_lw_gama  $train_epochs


#----------------------------------------
#aug_types='rpsfj-360_0_0'
#aug_types='psfj'
#optimizer='adam'
#feed_data='xyzrsg-nxnynz'
#----------------------------------------


