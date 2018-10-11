export CUDA_VISIBLE_DEVICES=1

net_flag='7A'
ng=1
bs=1
rs=--rs
#rs=''
normxyz='raw'

normedge='raw'
ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs   $rs --normxyz $normxyz --normedge $normedge
normedge='l0'
ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs   $rs --normxyz $normxyz --normedge $normedge


normxyz='mean0'

normedge='raw'
ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs   $rs --normxyz $normxyz --normedge $normedge
normedge='l0'
ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs   $rs --normxyz $normxyz --normedge $normedge


normxyz='min0'

normedge='raw'
ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs   $rs --normxyz $normxyz --normedge $normedge
normedge='l0'
ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs   $rs --normxyz $normxyz --normedge $normedge



#ply=--eval_only
ply=--pred_ply
model_dir='RC8B_bc8_120_02-xyz-n-adam-Drop000-Bs2-Lr1_6_20-wd4'
#ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs  $ply  $rs --model_dir $model_dir






