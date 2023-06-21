gpu=0
# gpu=0,1,2,3,4,5,6,7
# finetune_checkpoint="alpaca-lora/"
finetune_checkpoint="/home/v-lbei/alpaca-lora/lora-alpaca-13b-50k"
data_file=""
export CUDA_VISIBLE_DEVICES=$gpu
python3 generate_my.py --load_8bit --base_model ../alpaca13b --lora_weights $finetune_checkpoint