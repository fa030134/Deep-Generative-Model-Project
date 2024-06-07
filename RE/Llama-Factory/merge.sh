# 把lora模块和原模型merge到一起，得到微调后到新模型
python src/export_model.py \
    --model_name_or_path /home/sunhao/Llama2-Chinese-7b-Chat \
    --template default \
    --finetuning_type lora \
    --checkpoint_dir /home/sunhao/sclora/Llama-Factory/lora/checkpoint-2500 \
    --export_dir /home/sunhao/sclora/Llama-Factory/final_models/1