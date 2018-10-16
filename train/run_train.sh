#export CUDA_VISIBLE_DEVICES=1

net_flag='53A'
net_flag='8B'
ng=2
bs=2
rs=--rs
normxyz='min0'
normedge='raw'


ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs   $rs --normxyz $normxyz --normedge $normedge
#ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs  --normxyz $normxyz --normedge $normedge
#
#rs=--rs
#net_flag='53B'
#ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs   $rs --normxyz $normxyz --normedge $normedge
#ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs  --normxyz $normxyz --normedge $normedge
