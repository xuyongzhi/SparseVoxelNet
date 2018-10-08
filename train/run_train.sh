##--pred_ply
ply=--pred_ply
ply=''
#ply=--eval_only
net_flag='4B'

ng=1
bs=2

export CUDA_VISIBLE_DEVICES=0
ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs  $ply


