export CUDA_VISIBLE_DEVICES=1

net_flag='8A'
ng=1
bs=1
rs=--rs
normxyz='raw'
normedge='raw'

normxyz='min0'

ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs   $rs --normxyz $normxyz --normedge $normedge







