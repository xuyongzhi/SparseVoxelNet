#export CUDA_VISIBLE_DEVICES=1

net_flag='2A2'
ng=1
bs=3
rs=--rs
#normxyz='raw'
normxyz='min0'

normedge='raw'


ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs   $rs --normxyz $normxyz --normedge $normedge
#ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs   $rs --normxyz 'raw' --normedge $normedge
#ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs   --normxyz $normxyz --normedge $normedge
#ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs   $rs --normxyz $normxyz --normedge 'l0'
#
#
#net_flag='7B'
#ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs   $rs --normxyz $normxyz --normedge $normedge
#ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs   --normxyz $normxyz --normedge $normedge
#ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs   $rs --normxyz $normxyz --normedge 'l0'
#
#
#
