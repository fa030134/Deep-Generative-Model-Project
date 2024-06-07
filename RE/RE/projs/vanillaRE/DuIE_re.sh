DATA=DuIE
mode=test # or dev
#model=/home/LLMs/qwen/qwen/Qwen-7B-Chat
model=gpt-3.5-turbo
#chatglm3-6b
#Llama2-Chinese-7b-Chat
#Baichuan2-13B-Chat
#qwen/qwen/Qwen-14B-Chat
#qwen/qwen/Qwen-7B-Chat
test_subset_num=20000
# remove --debug for entire run
CUDA_VISIBLE_DEVICES=0 python vanilla_re_hf_llm.py --mode $mode --test_subset_samples $test_subset_num --dataset ${DATA} --ex_name vanilla --no_class_explain  --run_setting zero_shot --prompt_config_name vanilla_prompt_config.yaml --model $model --type_constrained
#"gpt-3.5-turbo"
#"gpt-4"