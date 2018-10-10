#export CUDA_VISIBLE_DEVICES=1

net_flag='8B'
ng=2
bs=2
rs=--rs
#rs=''

ply=''
ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs  $ply  $rs



#ply=--eval_only
ply=--pred_ply
model_dir='RC8B_bc8_120_02-xyz-n-adam-Drop000-Bs2-Lr1_6_20-wd4'
#ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs  $ply  $rs --model_dir $model_dir






