ply=''
#ply=--eval_only
#ply=--pred_ply
net_flag='8A'

ng=2
bs=2
rs=--rs
#rs=''

#export CUDA_VISIBLE_DEVICES=1
ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs  $ply  $rs


