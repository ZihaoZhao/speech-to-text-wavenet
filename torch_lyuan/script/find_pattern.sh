###
 # @Author       : Zihao Zhao
 # @E-mail       : zhzhao18@fudan.edu.cn
 # @Company      : Fudan University
 # @Date         : 2020-10-13 20:16:46
 # @LastEditors  : Zihao Zhao
 # @LastEditTime : 2020-11-04 14:39:06
 # @FilePath     : /speech-to-text-wavenet/torch_lyuan/script/find_pattern.sh
 # @Description  : 
### 
#!/usr/bin/env bash

#/zhzhao/miniconda3/bin/conda init bash | conda activate pytorch16 | bash /zhzhao/code/wavenet_torch/torch_lyuan/script/find_pattern.sh
exp_name="pruning_from_dense_color"
excel_name=$exp_name".xls"
model_name="/zhzhao/code/wavenet_torch/torch_lyuan/exp_result/best.pth"

sparsity_list=("0.05" \
                "0.10" \
                "0.15" \
                "0.20" \
                "0.25" \
                "0.30" \
                "0.40" \
                "0.50" \
                "0.60" \
                "0.65" \
                "0.70" \
                "0.75" \
                "0.80" \
                "0.85" \
                "0.90" \
                "0.95")

coonum_list=("1" \
            "2" \
            "4" \
            "6" \
            "8" \
            "10" \
            "12" \
            "16" \
            "20" \
            "24" \
            "28" \
            "32" \
            "36" \
            "40" \
            "44" \
            "48" \
            "52" \
            "56" \
            "60" \
            "64" \
            "68" \
            "72" \
            "76" \
            "84" \
            "88" \
            "100")

pattern_shape_list=("16_16" \
                "8_8" \
                "4_4" \
                "4_8" \
                "4_16" \
                "8_16")

for pattern_shape in ${pattern_shape_list[*]}
do
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
            --find_pattern_shape $pattern_shape \
            --save_pattern_count_excel $excel_name
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

