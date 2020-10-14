using distributed to speed up training
# how to run
1. setting dataset path and other config in config_train.py
2. fix node numbers and the path of train_distributed.py in dist_train.sh
3. bash dist_train.sh



ps -ef|grep python|cut -c 9-15 |xargs kill -s9

CUDA_VISIBLE_DEVICES=0 screen python train.py --exp dense_32

CUDA_VISIBLE_DEVICES=1 screen python train.py --exp sparse_020_32 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/dense_32/debug/weights/last.pth --sparse_mode sparse_pruning --sparsity 0.2

CUDA_VISIBLE_DEVICES=0 screen python train.py --exp sparse_050_32 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/dense_32/debug/weights/last.pth --sparse_mode sparse_pruning --sparsity 0.5

CUDA_VISIBLE_DEVICES=1 screen python train.py --exp sparse_080_32 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/dense_32/debug/weights/last.pth --sparse_mode sparse_pruning --sparsity 0.8