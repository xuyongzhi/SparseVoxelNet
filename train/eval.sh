export CUDA_VISIBLE_DEVICES=1


#ply=--eval_only
ply=--pred_ply
model_dir='RC10G_12_6_bc12_12_69_2-xyz-n-min0-Bs2-Lr0_7_20-wd0'

net_flag='10G_12_6'
ng=1
bs=1
rs=--rs


ipython  ./train_main.py -- --net_flag $net_flag --num_gpus $ng --batch_size $bs   $rs  $ply --model_dir $model_dir
