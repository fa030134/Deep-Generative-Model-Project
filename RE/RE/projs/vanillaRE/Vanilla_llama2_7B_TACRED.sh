# /home/LLMs/EleutherAI/gpt-neox-20b
# /home/LLMs/llama-2-7b-chat
# remove --debug for entire run
# in the vanillaRE folder
DATA=RETACRED
mode=test # or dev
model=/home/LLMs/llama-2-7b-chat

# remove --debug for series run
CUDA_VISIBLE_DEVICES=0 python vanilla_re_hf_llm.py --mode $mode --no_class_explain --ex_name vanilla --dataset ${DATA} --type_constrained --run_setting zero_shot --prompt_config_name vanilla_prompt_config.yaml --model $model

# saved file in '../../outputs/{}/{}/{}/{}'.format(args.dataset, args.ex_name, args.engine.replace('/', '-'), args.run_setting)
