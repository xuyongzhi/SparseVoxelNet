#export CUDA_VISIBLE_DEVICES=0

net_flag='4A'
ng=2
bs=2
rs=--rs


ipython  ./train_main.py -- --net_flag $net_flag --num_gpus $ng --batch_size $bs   $rs 

