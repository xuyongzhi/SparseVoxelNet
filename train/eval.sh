export CUDA_VISIBLE_DEVICES=1


ply=--eval_only
#ply=--pred_ply
model_dir='RC1A_bc-xyz-n-min0-Bs32-Lr0_10_20-wd0'

ng=1
rs=--rs

sg='8192_1'
bs=32
normxyz='min0'

net_flag='1A'

ipython  ./train_main.py -- --net_flag $net_flag --num_gpus $ng --batch_size $bs   $rs  --sg $sg --normxyz $normxyz  $ply --model_dir $model_dir

