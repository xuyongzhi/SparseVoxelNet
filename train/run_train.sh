#export CUDA_VISIBLE_DEVICES=1

net_flag='8A'
ng=2
bs=2
rs=--rs
normxyz='raw'
#normxyz='min0'

normedge='raw'


ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs   $rs --normxyz $normxyz --normedge $normedge
ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs   --normxyz $normxyz --normedge $normedge
ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs   $rs --normxyz $normxyz --normedge $normedge --nobn
ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs   $rs --normxyz $normxyz --normedge 'l0'
ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs   $rs --normxyz $normxyz --normedge 'l0' --nobn

ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs   $rs --normxyz $normxyz --normedge $normedge --act Lrelu







