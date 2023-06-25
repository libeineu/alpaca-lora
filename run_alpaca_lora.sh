gpu=0
# gpu=0,1,2,3,4,5,6,7
# finetune_checkpoint="alpaca-lora/"
finetune_checkpoint=/home/v-lbei/alpaca-lora/625_7b
data_file=""
export CUDA_VISIBLE_DEVICES=$gpu
python3 generate_my.py --load_8bit --base_model chavinlo/alpaca-native --lora_weights $finetune_checkpoint