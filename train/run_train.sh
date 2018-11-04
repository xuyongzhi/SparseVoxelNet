#export CUDA_VISIBLE_DEVICES=0

ng=2
rs=--rs

sg='8192_1'
bs=36
normxyz='min0'
net_flag='1A'
lw_delta=0
wd='0'
drop_imo='005'

ipython  ./train_main.py -- --net_flag $net_flag --num_gpus $ng --batch_size $bs   $rs  --sg $sg --normxyz $normxyz --lwgama $lwgama --wd $wd --drop_imo $drop_imo

#ipython  ./train_main.py -- --net_flag $net_flag --num_gpus $ng --batch_size $bs   $rs  --sg $sg --normxyz $normxyz --lwgama '1.2' --wd $wd --drop_imo $drop_imo
#
#
#wd='1e-4'
#drop_imo='000'
#
#ipython  ./train_main.py -- --net_flag $net_flag --num_gpus $ng --batch_size $bs   $rs  --sg $sg --normxyz $normxyz --lwgama $lwgama --wd $wd --drop_imo $drop_imo

