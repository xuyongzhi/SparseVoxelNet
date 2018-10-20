export CUDA_VISIBLE_DEVICES=0

net_flag='7A_9_5'
net_flag='2G'
#net_flag='7G_9_5'
ng=1
bs=1
rs=--rs


ipython  ./train_main.py -- --net_flag $net_flag --num_gpus $ng --batch_size $bs   $rs 

