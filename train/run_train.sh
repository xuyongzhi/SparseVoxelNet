##--pred_ply

export CUDA_VISIBLE_DEVICES=0
python ./train_main.py --num_gpus 1 --wd 1e-4  --pred_ply

export CUDA_VISIBLE_DEVICES=1
#python ./train_main.py --num_gpus 1 --wd 0 
