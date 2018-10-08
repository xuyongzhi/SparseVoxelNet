##--pred_ply
wd=1e-4
ply=--pred_ply

export CUDA_VISIBLE_DEVICES=1
python ./train_main.py --num_gpus 1 --wd 1e-4 $ply
#python ./train_main.py --num_gpus 1 --wd $wd

