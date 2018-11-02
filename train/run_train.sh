#export CUDA_VISIBLE_DEVICES=0

ng=2
rs=--rs

sg='8192_1'
net_flag='1B'
bs=32
normxyz='min0'

#normxyz='raw'
net_flag='1A'

ipython  ./train_main.py -- --net_flag $net_flag --num_gpus $ng --batch_size $bs   $rs  --sg $sg --normxyz $normxyz
