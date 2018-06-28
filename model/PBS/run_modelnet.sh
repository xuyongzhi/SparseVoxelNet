
batch_size=32
model_flag='m'
learning_rate0=0.001
lr_decay_epochs=10
lr_decay_rate=0.7
num_gpus=2
feed_data='xyzs-nxnynz'
drop_imo='0_0_5'
num_filters0=32
optimizer='momentum'
aug_types='N'
use_bias=1

#aug_types='r-360_0_0'

./modelnet.sh   $aug_types  $batch_size $model_flag $learning_rate0 $num_gpus $feed_data $drop_imo $num_filters0 $optimizer $use_bias $lr_decay_epochs $lr_decay_rate



#num_filters0=64
#./modelnet.sh   $aug_types  $batch_size $model_flag $learning_rate0 $num_gpus $feed_data $drop_imo $num_filters0 $optimizer $use_bias $lr_decay_epochs $lr_decay_rate
#num_filters0=32
#
#use_bias=0
#./modelnet.sh   $aug_types  $batch_size $model_flag $learning_rate0 $num_gpus $feed_data $drop_imo $num_filters0 $optimizer $use_bias $lr_decay_epochs $lr_decay_rate

#----------------------------------------
#aug_types='rpsfj-360_0_0'
#aug_types='psfj'
#optimizer='adam'
#feed_data='xyzrsg-nxnynz'
#----------------------------------------



