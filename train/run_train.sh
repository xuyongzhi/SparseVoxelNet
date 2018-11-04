#export CUDA_VISIBLE_DEVICES=0

ng=2
rs=--rs

sg='8192_1'
bs=2
normxyz='min0'
net_flag='1A'
lwgama=2
wd='1e-4'

ipython  ./train_main.py -- --net_flag $net_flag --num_gpus $ng --batch_size $bs   $rs  --sg $sg --normxyz $normxyz --lwgama $lwgama --wd $wd

#normxyz='mean0'
#ipython  ./train_main.py -- --net_flag $net_flag --num_gpus $ng --batch_size $bs   $rs  --sg $sg --normxyz $normxyz
