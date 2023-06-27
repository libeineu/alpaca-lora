gpu=0
# gpu=0,1,2,3,4,5,6,7
# finetune_checkpoint="alpaca-lora/"
finetune_checkpoint=Angainor/alpaca-lora-13b
data_file=""
export CUDA_VISIBLE_DEVICES=$gpu
# python3 generate_my.py --load_8bit --base_model ../alpaca13b --lora_weights $finetune_checkpoint
python3 generate_my.py --load_8bit --base_model 'decapoda-research/llama-13b-hf' --lora_weights $finetune_checkpoint
