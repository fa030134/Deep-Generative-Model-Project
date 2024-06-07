DATA=DuIE
mode=test # or dev
#model=/home/LLMs/qwen/qwen/Qwen-14B-Chat
model=/home/sunhao/sclora/Llama-Factory/final_models/1
# chatglm3-6b
# Llama2-Chinese-7b-Chat
# Baichuan2-13B-Chat
# qwen/qwen/Qwen-7B-Chat
# qwen/qwen/Qwen-14B-Chat
# remove --debug for entire run
#"gpt-3.5-turbo"
#"gpt-4"
testnum=18000
CUDA_VISIBLE_DEVICES=3 python qa4re_hf_llm.py --test_subset_samples $testnum --mode $mode --dataset ${DATA} --run_setting zero_shot --prompt_config_name qa4re_prompt_config.yaml --model $model #--type_constrained #--debug