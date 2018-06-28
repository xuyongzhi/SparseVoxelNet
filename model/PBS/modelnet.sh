resnet_size=28
residual=1
batch_norm_decay0=0.7

modelnet()
{
  aug_types=$1
  batch_size=$2
  model_flag=$3
  learning_rate0=$4
  num_gpus=$5
  feed_data=$6
  drop_imo=$7
  num_filters0=$8
  optimizer=$9
  use_bias=${10}
  lr_decay_epochs=${11}
  lr_decay_rate=${12}
  python ../modelnet_main.py  --resnet_size $resnet_size --model_flag $model_flag --num_gpus 2 --batch_size $batch_size --feed_data $feed_data --aug_types $aug_types --learning_rate0 $learning_rate0 --optimizer $optimizer --batch_norm_decay0 $batch_norm_decay0 --learning_rate0 $learning_rate0 --num_gpus $num_gpus --drop_imo $drop_imo --num_filters0 $num_filters0 --residual $residual --use_bias $use_bias --lr_decay_epochs $lr_decay_epochs --lr_decay_rate $lr_decay_rate
}

modelnet $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12}
