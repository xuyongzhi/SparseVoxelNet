##--pred_ply
ply=--pred_ply
ply=''
#ply=--eval_only
net_flag='5A'

ng=2
bs=4
rs=--rs

#export CUDA_VISIBLE_DEVICES=1
ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs  $ply  $rs


