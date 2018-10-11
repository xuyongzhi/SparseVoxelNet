#export CUDA_VISIBLE_DEVICES=1

net_flag='5A'
ng=2
bs=4
rs=--rs
#rs=''
normxyz='raw'
#normxyz='mean0'
#normxyz='min0'
#normxyz='max1'

normedge='l0'
ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs   $rs --normxyz $normxyz --normedge $normedge

normedge='all'
ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs   $rs --normxyz $normxyz --normedge $normedge


normxyz='mean0'
normedge='l0'
ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs   $rs --normxyz $normxyz --normedge $normedge

normedge='all'
ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs   $rs --normxyz $normxyz --normedge $normedge


#ply=--eval_only
ply=--pred_ply
model_dir='RC8B_bc8_120_02-xyz-n-adam-Drop000-Bs2-Lr1_6_20-wd4'
#ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs  $ply  $rs --model_dir $model_dir






