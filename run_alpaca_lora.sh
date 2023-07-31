gpu=0
# gpu=0,1,2,3,4,5,6,7
# finetune_checkpoint="alpaca-lora/"
# finetune_checkpoint=output/llama-13b-fixed-instruction-new
finetune_checkpoint=output/wmt21_50K
data_file=""
export CUDA_VISIBLE_DEVICES=$gpu
# python3 generate_my.py --load_8bit --base_model ../alpaca13b --lora_weights $finetune_checkpoint
# python3 generate_my.py --load_8bit --base_model 'decapoda-research/llama-7b-hf' 

python3 generate_batch.py --load_8bit --base_model ../alpaca13b --lora_weights $finetune_checkpoint --batch_size 4
# python3 generate_batch.py --load_8bit --base_model 'decapoda-research/llama-7b-hf' --lora_weights 'tloen/alpaca-lora' --batch_size 4
