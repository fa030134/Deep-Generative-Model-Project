# 如果不需要开wandb记录训练过程信息的话，可以 export WANDB_DISABLED=true
CUDA_VISIBLE_DEVICES=3 accelerate launch --config_file lora_config.yaml src/train_bash.py \
    --stage sft \
    --model_name_or_path /LLMs/Llama2-Chinese-7b-Chat \
    --do_train \
    --dataset info_train_2000 \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --template default \
    --output_dir /home/sunhao/sclora/Llama-Factory/lora \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 20 \
    --learning_rate 5e-5 \
    --num_train_epochs 20 \
    --plot_loss \
    --fp16 \
    --seed 0