export CUDA_VISIBLE_DEVICES=1

net_flag='5A'
ng=1
bs=1
rs=--rs
#rs=''

ply=''
ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs  $ply  $rs



ply=--eval_only
#ply=--pred_ply
model_dir='RC8A_bc8_120_22-xyz-n-adam-Drop000-Bs2-Lr1_7_30-wd4'
#ipython  ./train_main.py -- --num_gpus $ng --net_flag $net_flag --batch_size $bs  $ply  $rs --model_dir $model_dir






