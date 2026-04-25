#!/bin/bash

model="meta-llama/Llama-3.2-1B-Instruct"
#model="Qwen/Qwen3.5-0.8B"
#model="google/gemma-3-1b-it"
data_path="/A/thomas/TRACE-Benchmark/LLM-CL-Benchmark_500"
#inference_tasks="C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten"
inference_tasks="C-STANCE,FOMC"
cl_method="lora"

output_root="/A/thomas/TRACE-Benchmark/outputs/LLM-CL_Benchmark_500"
output_dir="${output_root}/${cl_method}/${model//\//_}/"
inference_model_path="$output_dir"
inference_output_path="${output_dir}/predictions"

mkdir -p "$output_dir" "$inference_output_path"

port=$(shuf -i25000-30000 -n1)

uv run deepspeed --master_port "$port" inference/infer_single.py \
   --data_path "$data_path" \
   --inference_tasks "$inference_tasks" \
   --model_name_or_path "$model" \
   --inference_model_path "$inference_model_path" \
   --inference_batch 4 \
   --max_prompt_len 1024 \
   --max_ans_len 512 \
   --seed 1234 \
   --deepspeed \
   --CL_method "$cl_method" \
   --inference_output_path "$inference_output_path" > "${output_dir}/infer.log" 2>&1 &