#!/bin/bash

# 定义要遍历的目录
DIRECTORY="./deen/test/"

# 切换到该目录
cd $DIRECTORY

# 创建8个子目录（如果它们不存在）
for i in {1..8}; do
    mkdir -p "folder$i"
done

# 遍历目录下的每个文件
for file in *; do
    # 如果当前项是文件而不是目录
    if [ -f "$file" ]; then
        total_lines=$(wc -l < "$file")

        # 如果文件行数可以被8整除
        if [ $(($total_lines % 8)) -eq 0 ]; then
            # 计算每个子文件应该有的行数
            lines_per_file=$(($total_lines / 8))

            # 使用split命令明确指定每个子文件的行数
            split -l $lines_per_file "$file" "$file-"

            # 将分割后的文件移动到对应的子目录
            mv "${file}-aa" folder1/$file
            mv "${file}-ab" folder2/$file
            mv "${file}-ac" folder3/$file
            mv "${file}-ad" folder4/$file
            mv "${file}-ae" folder5/$file
            mv "${file}-af" folder6/$file
            mv "${file}-ag" folder7/$file
            mv "${file}-ah" folder8/$file
        else
            echo "File $file cannot be evenly divided into 8 parts. Skipping..."
        fi
    fi
done

cd /mnt/zhengtong/alpaca-lora

# 为每一个GPU设定一个唯一ID (例如：0,1,2,...)
# 循环从0到7
for GPU_ID in {0..7}; do
    # 使用CUDA_VISIBLE_DEVICES指定要使用的GPU，并运行PyTorch脚本
    # 这假设你的PyTorch代码已经配置为使用设备'cuda'
    num=$((GPU_ID + 1))
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 vicuna_generate_pink.py --test_file=deen/test/folder$num --output_file=output/PINK$num &> log/generate_$num.txt &
done

exit 0
