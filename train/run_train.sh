##--pred_ply
ply=--pred_ply
ply=''
#ply=--eval_only
net_flag='5A'

export CUDA_VISIBLE_DEVICES=1
ipython  ./train_main.py -- --num_gpus 1 --net_flag $net_flag  $ply

