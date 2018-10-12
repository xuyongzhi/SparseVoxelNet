#export CUDA_VISIBLE_DEVICES=1

net_flag='8A'
ng=2
bs=2
rs=--rs
normxyz='raw'
normedge='raw'


ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs   $rs --normxyz $normxyz --normedge $normedge

normxyz='min0'
#ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs   $rs --normxyz $normxyz --normedge $normedge






