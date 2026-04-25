#!/bin/bash

model="meta-llama/Llama-3.2-1B-Instruct"
#model="Qwen/Qwen3.5-0.8B"
#model="google/gemma-3-1b-it"
data_path="/A/thomas/TRACE-Benchmark/LLM-CL-Benchmark_500"
dataset_name="C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten"
#dataset_name="C-STANCE,FOMC"
cl_method="lora"

output_root="/A/thomas/TRACE-Benchmark/outputs/LLM-CL_Benchmark_500_all"
output_dir="${output_root}/${cl_method}/${model//\//_}/"

mkdir -p "$output_dir"

port=$(shuf -i25000-30000 -n1)

uv run deepspeed --master_port "$port" training/main.py \
   --data_path "$data_path" \
   --dataset_name "$dataset_name" \
   --model_name_or_path "$model" \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 16 \
   --max_prompt_len 1024 \
   --max_ans_len 512 \
   --learning_rate 1e-4 \
   --weight_decay 0. \
   --num_train_epochs 5,3,7,5,3,5,5,7 \
   --gradient_accumulation_steps 2 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage 2 \
   --deepspeed \
   --print_loss \
   --CL_method "$cl_method" \
   --output_dir "$output_dir" > "${output_dir}/train.log" 2>&1 &