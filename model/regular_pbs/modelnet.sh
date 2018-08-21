epochs_between_evals=1

modelnet()
{
  aug_types=$1
  batch_size=$2
  model_flag=$3
  learning_rate0=$4
  num_gpus=$5
  feed_data=$6
  drop_imo=$7
  optimizer=$8
  use_bias=${9}
  lr_decay_epochs=${10}
  lr_decay_rate=${11}
  resnet_size=${12}
  block_style=${13}
  residual=${14}
  shortcut=${15}
  loss_lw_gama=${16}
  train_epochs=${17}
  batch_norm_decay0=${18}
  gpu_id=${19}
  use_xyz=${20}

  if [ $num_gpus == 1 ]
  then
    export CUDA_VISIBLE_DEVICES=$gpu_id
  fi

  python ../modelnet_main.py  --resnet_size $resnet_size --model_flag $model_flag --num_gpus 2 --batch_size $batch_size --feed_data $feed_data --aug_types $aug_types --learning_rate0 $learning_rate0 --optimizer $optimizer --batch_norm_decay0 $batch_norm_decay0 --learning_rate0 $learning_rate0 --num_gpus $num_gpus --drop_imo $drop_imo --residual $residual --use_bias $use_bias --lr_decay_epochs $lr_decay_epochs --lr_decay_rate $lr_decay_rate --block_style $block_style --shortcut $shortcut --loss_lw_gama $loss_lw_gama --train_epochs $train_epochs --gpu_id $gpu_id --epochs_between_evals $epochs_between_evals --use_xyz $use_xyz
}

modelnet $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17} ${18} ${19} ${20}
