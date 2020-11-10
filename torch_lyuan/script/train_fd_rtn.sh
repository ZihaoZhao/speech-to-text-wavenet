###
 # @Author       : Zihao Zhao
 # @E-mail       : zhzhao18@fudan.edu.cn
 # @Company      : Fudan University
 # @Date         : 2020-10-13 20:16:46
 # @LastEditors  : Zihao Zhao
 # @LastEditTime : 2020-11-10 23:00:24
 # @FilePath     : /speech-to-text-wavenet/torch_lyuan/script/train_fd_rtn.sh
 # @Description  : 
### 
#!/usr/bin/env bash
# /zhzhao/miniconda3/bin/conda init bash | conda activate pytorch16 | /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp fd_rtn_16_2_2_2 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --sparse_mode find_retrain --find_retrain_para 16_2_2_2 --lr 0.001  --save_excel find_retrain.xls
# /zhzhao/miniconda3/bin/conda init bash | conda activate pytorch16 | /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp fd_rtn_16_2_4_2 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --sparse_mode find_retrain --find_retrain_para 16_2_4_2 --lr 0.001  --save_excel find_retrain.xls
# /zhzhao/miniconda3/bin/conda init bash | conda activate pytorch16 | /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp fd_rtn_16_4_4_2 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --sparse_mode find_retrain --find_retrain_para 16_4_4_2 --lr 0.001  --save_excel find_retrain.xls
# /zhzhao/miniconda3/bin/conda init bash | conda activate pytorch16 | /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp fd_rtn_16_4_8_2 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --sparse_mode find_retrain --find_retrain_para 16_4_8_2 --lr 0.001  --save_excel find_retrain.xls
# /zhzhao/miniconda3/bin/conda init bash | conda activate pytorch16 | /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp fd_rtn_16_8_8_2 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --sparse_mode find_retrain --find_retrain_para 16_8_8_2 --lr 0.001  --save_excel find_retrain.xls
# /zhzhao/miniconda3/bin/conda init bash | conda activate pytorch16 | /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp fd_rtn_16_8_16_2 --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --sparse_mode find_retrain --find_retrain_para 16_8_16_2 --lr 0.001  --save_excel find_retrain.xls

#/zhzhao/miniconda3/bin/conda init bash | conda activate pytorch16 | bash /zhzhao/code/wavenet_torch/torch_lyuan/script/train_fd_rtn.sh
model_name="/zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth"

nnz_list=("2")
pattern_num_list=("16")

pattern_shape_list=("2_2" \
                    "2_4" \
                    "4_4" \
                    "4_8" \
                    "8_8" \
                    "8_16")


for pattern_num in ${pattern_num_list[*]}
do
    for nnz in ${nnz_list[*]}
    do
        for pattern_shape in ${pattern_shape_list[*]}
        do
            /zhzhao/miniconda3/envs/pytorch16/bin/python \
            /zhzhao/code/wavenet_torch/torch_lyuan/train.py \
            --exp "fd_rtn_"$pattern_num"_"$pattern_shape"_"$nnz \
            --load_from $model_name \
            --sparse_mode find_retrain \
            --find_retrain_para $pattern_num"_"$pattern_shape"_"$nnz \
            --lr 0.001  --save_excel find_retrain.xls
        done
    done
done

# /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp dense --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --find_pattern --find_pattern_para 0.01_64
# /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp dense --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --find_pattern --find_pattern_para 0.02_64
# /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp dense --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --find_pattern --find_pattern_para 0.04_64
# /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp dense --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --find_pattern --find_pattern_para 0.01_32
# /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp dense --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --find_pattern --find_pattern_para 0.02_32
# /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp dense --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --find_pattern --find_pattern_para 0.04_32
# /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp dense --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --find_pattern --find_pattern_para 0.01_16
# /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp dense --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --find_pattern --find_pattern_para 0.02_16
# /zhzhao/miniconda3/envs/pytorch16/bin/python /zhzhao/code/wavenet_torch/torch_lyuan/train.py --exp dense --load_from /zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth --find_pattern --find_pattern_para 0.04_16

