export CUDA_VISIBLE_DEVICES=1


#ply=--eval_only
ply=--pred_ply
model_dir='RC7G_9_5_bc9_9_85_2-xyz-n-min0-Bs2-Lr1_7_20-wd4'

net_flag='7G_9_5'
ng=1
bs=1
rs=--rs


ipython  ./train_main.py -- --net_flag $net_flag --num_gpus $ng --batch_size $bs   $rs  $ply --model_dir $model_dir
