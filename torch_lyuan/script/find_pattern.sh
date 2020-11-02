###
 # @Author       : Zihao Zhao
 # @E-mail       : zhzhao18@fudan.edu.cn
 # @Company      : Fudan University
 # @Date         : 2020-10-13 20:16:46
 # @LastEditors  : Zihao Zhao
 # @LastEditTime : 2020-11-02 14:06:00
 # @FilePath     : /speech-to-text-wavenet/torch_lyuan/train_script/find_pattern.sh
 # @Description  : 
### 
#!/usr/bin/env bash

#/zhzhao/miniconda3/bin/conda init bash | conda activate pytorch16 | sh /zhzhao/code/wavenet_torch/torch_lyuan/train_script/find_pattern.sh
exp_name="dense_pattern_curve_test6"
excel_name=$exp_name".xls"
model_name="/zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth"

sparsity_list=("0.25" \
                "0.5" \
                "0.75")

coonum_list=("16" \
            "32" \
            "48" \
            "64")

# TODO more data
pattern_shape

for sparsity in ${sparsity_list[*]}
do
    for coonum in ${coonum_list[*]}
    do
        /zhzhao/miniconda3/envs/pytorch16/bin/python \
        /zhzhao/code/wavenet_torch/torch_lyuan/train.py \
        --exp $exp_name \
        --load_from $model_name \
        --find_pattern \
        --find_pattern_para $sparsity"_"$coonum \
        --save_pattern_count_excel $excel_name
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

