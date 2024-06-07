# /home/LLMs/EleutherAI/gpt-neox-20b
# /home/LLMs/llama-2-7b-chat
DATA=TACREV
mode=test # or dev
model=/home/LLMs/EleutherAI/gpt-neox-20b
# remove --debug for entire run
CUDA_VISIBLE_DEVICES=3 python qa4re_hf_llm.py --mode $mode --dataset ${DATA} --run_setting zero_shot --type_constrained --prompt_config_name qa4re_prompt_config.yaml --model $model
