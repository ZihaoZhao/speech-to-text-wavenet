<!--
 * @Author: your name
 * @Date: 2020-10-17 18:33:49
 * @LastEditTime : 2020-10-21 14:54:07
 * @LastEditors  : Zihao Zhao
 * @Description: In User Settings Edit
 * @FilePath     : /speech-to-text-wavenet/torch_lyuan/readme.md
-->
using distributed to speed up training
# how to run
1. setting dataset path and other config in config_train.py
2. fix node numbers and the path of train_distributed.py in dist_train.sh
3. bash dist_train.sh

# ctc decoder
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install .

# run exp

```
ps -ef|grep python|cut -c 9-15 |xargs kill -s9

```

# exp pruning

```
/zhzhao/miniconda3/bin/conda init | conda activate pytorch16 | /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp dense --sparse_mode dense --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/dense_32_001_1212_best.pth

/zhzhao/miniconda3/bin/conda init | conda activate pytorch16 | /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp dense --sparse_mode dense --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/dense_32_001_1212_best.pth

/zhzhao/miniconda3/bin/conda init | conda activate pytorch16 | /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp sparse_25_step --sparse_mode sparse_pruning --sparsity 0.25 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/dense_32_001_1212_best.pth

/zhzhao/miniconda3/bin/conda init | conda activate pytorch16 | /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp sparse_50_step --sparse_mode sparse_pruning --sparsity 0.5 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/dense_32_001_1212_best.pth

/zhzhao/miniconda3/bin/conda init | conda activate pytorch16 | /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp sparse_75_step --sparse_mode sparse_pruning --sparsity 0.75 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/dense_32_001_1212_best.pth
```


# exp pattern_num-sparsity
```
/zhzhao/miniconda3/bin/conda init | conda activate pytorch16 | /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_16_1616_128 --sparse_mode pattern_pruning --pattern_para 16_16_16_128 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/dense_32_001_1212_best.pth

/zhzhao/miniconda3/bin/conda init | conda activate pytorch16 | /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_8_1616_128 --sparse_mode pattern_pruning --pattern_para 8_16_16_128 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/dense_32_001_1212_best.pth

/zhzhao/miniconda3/bin/conda init | conda activate pytorch16 | /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_4_1616_128 --sparse_mode pattern_pruning --pattern_para 4_16_16_128 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/dense_32_001_1212_best.pth

/zhzhao/miniconda3/bin/conda init | conda activate pytorch16 | /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_1_1616_128 --sparse_mode pattern_pruning --pattern_para 1_16_16_128 --load_from /mnt/inspurfs/user-fs/zhzhao/code/wavenet_torch/torch_lyuan/exp_result/dense/debug/weights/best.pth

/zhzhao/miniconda3/bin/conda init | conda activate pytorch16 | /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_16_1616_192 --sparse_mode pattern_pruning --pattern_para 16_16_16_192 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/dense_32_001_1212_best.pth

/zhzhao/miniconda3/bin/conda init | conda activate pytorch16 | /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_8_1616_192 --sparse_mode pattern_pruning --pattern_para 8_16_16_192 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/dense_32_001_1212_best.pth

/zhzhao/miniconda3/bin/conda init | conda activate pytorch16 | /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_4_1616_192 --sparse_mode pattern_pruning --pattern_para 4_16_16_192 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/dense_32_001_1212_best.pth

/zhzhao/miniconda3/bin/conda init | conda activate pytorch16 | /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_1_1616_192 --sparse_mode pattern_pruning --pattern_para 1_16_16_192 --load_from /mnt/inspurfs/user-fs/zhzhao/code/wavenet_torch/torch_lyuan/exp_result/dense/debug/weights/best.pth

/zhzhao/miniconda3/bin/conda init | conda activate pytorch16 | /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_16_1616_64 --sparse_mode pattern_pruning --pattern_para 16_16_16_64 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/dense_32_001_1212_best.pth

/zhzhao/miniconda3/bin/conda init | conda activate pytorch16 | /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_8_1616_64 --sparse_mode pattern_pruning --pattern_para 8_16_16_64 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/dense_32_001_1212_best.pth

/zhzhao/miniconda3/bin/conda init | conda activate pytorch16 | /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_4_1616_64 --sparse_mode pattern_pruning --pattern_para 4_16_16_64 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/dense_32_001_1212_best.pth

/zhzhao/miniconda3/bin/conda init | conda activate pytorch16 | /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_1_1616_64 --sparse_mode pattern_pruning --pattern_para 1_16_16_64 --load_from /mnt/inspurfs/user-fs/zhzhao/code/wavenet_torch/torch_lyuan/exp_result/dense/debug/weights/best.pth
```


# exp coo-sparsity
```
/zhzhao/miniconda3/bin/conda init | conda activate pytorch16 | /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_32 --sparse_mode coo_pruning --coo_para 8_8_32 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/dense_32_001_1212_best.pth

/zhzhao/miniconda3/bin/conda init | conda activate pytorch16 | /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_16 --sparse_mode coo_pruning --coo_para 8_8_16 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/dense_32_001_1212_best.pth

/zhzhao/miniconda3/bin/conda init | conda activate pytorch16 | /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp coo_8_8_48 --sparse_mode coo_pruning --coo_para 8_8_48 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/dense_32_001_1212_best.pth
```

# exp ptcoo-sparsity
```
/zhzhao/miniconda3/bin/conda init | conda activate pytorch16 | /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp ptcoo_16_1616_128_64 --sparse_mode ptcoo_pruning --ptcoo_para 16_16_16_128_64 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/dense_32_001_1212_best.pth

/zhzhao/miniconda3/bin/conda init | conda activate pytorch16 | /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp ptcoo_16_1616_80_48 --sparse_mode ptcoo_pruning --ptcoo_para 16_16_16_80_48 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/dense_32_001_1212_best.pth

/zhzhao/miniconda3/bin/conda init | conda activate pytorch16 | /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp ptcoo_16_1616_32_32 --sparse_mode ptcoo_pruning --ptcoo_para 16_16_16_32_32 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/dense_32_001_1212_best.pth
```

# exp visualiztion
```
vis_mask: /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_16_1616_64 --resume --vis_mask

vis_pattern: /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp pt_16_1616_64 --resume --vis_pattern
```

# former
```
CUDA_VISIBLE_DEVICES=0 screen python train.py --exp dense_32

CUDA_VISIBLE_DEVICES=1 screen python train.py --exp sparse_020_32 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/dense_32/debug/weights/last.pth --sparse_mode sparse_pruning --sparsity 0.2

CUDA_VISIBLE_DEVICES=0 screen python train.py --exp sparse_050_32 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/dense_32/debug/weights/last.pth --sparse_mode sparse_pruning --sparsity 0.5

CUDA_VISIBLE_DEVICES=1 screen python train.py --exp sparse_080_32 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/dense_32/debug/weights/last.pth --sparse_mode sparse_pruning --sparsity 0.8
```