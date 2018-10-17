#export CUDA_VISIBLE_DEVICES=1

net_flag='7A_8_5'
#net_flag='7G_8_5'
ng=2
bs=2
rs=--rs
normxyz='min0'
normedge='raw'


ipython  ./train_main.py -- --net_flag '7A_10_6' --num_gpus $ng --batch_size $bs   $rs --normxyz $normxyz --normedge $normedge
#ipython  ./train_main.py -- --net_flag '7A_12_6' --num_gpus $ng --batch_size $bs   $rs --normxyz $normxyz --normedge $normedge
#ipython  ./train_main.py -- --net_flag '7A_8_5' --num_gpus $ng --batch_size $bs   $rs --normxyz $normxyz --normedge $normedge --nobn
#ipython  ./train_main.py -- --net_flag '7A_12_6' --num_gpus $ng --batch_size $bs   $rs --normxyz $normxyz --normedge $normedge --nobn

#ipython  ./train_main.py -- --net_flag '7G_8_5' --num_gpus $ng --batch_size $bs   $rs --normxyz $normxyz --normedge $normedge
#ipython  ./train_main.py -- --net_flag '7G_12_6' --num_gpus $ng --batch_size $bs   $rs --normxyz $normxyz --normedge $normedge
#
#ipython  ./train_main.py -- --net_flag '7G_8_5' --num_gpus $ng --batch_size $bs   $rs --normxyz $normxyz --normedge $normedge --nobn
#ipython  ./train_main.py -- --net_flag '7G_12_6' --num_gpus $ng --batch_size $bs   $rs --normxyz $normxyz --normedge $normedge --nobn

