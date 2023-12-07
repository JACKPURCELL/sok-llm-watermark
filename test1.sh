#!/bin/bash
# CUDA_VISIBLE_DEVICES=1 python main.py --watermark xuandong23b --prompt_file /home/jkl6486/sok-llm-watermark/dataset/1000.jsonl > runs/xuandong23b.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python main.py --watermark lean23 --prompt_file /home/jkl6486/sok-llm-watermark/dataset/1000.jsonl > runs/lean23.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python main.py --watermark john23 --prompt_file /home/jkl6486/sok-llm-watermark/dataset/1000.jsonl > runs/john23.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=1 python main.py --watermark xuandong23b --attack dipper --skip_inject True --prompt_file /home/jkl6486/sok-llm-watermark/dataset/1000.jsonl > runs/xuandong23b_attack.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1 python main.py --watermark lean23 --attack dipper --skip_inject True --prompt_file /home/jkl6486/sok-llm-watermark/dataset/1000.jsonl > runs/lean23_attack.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python main.py --watermark john23 --attack dipper --skip_inject True --prompt_file /home/jkl6486/sok-llm-watermark/dataset/1000.jsonl > runs/john23_attack.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=1  python main.py --watermark rohith23  --prompt_file /home/jkl6486/sok-llm-watermark/dataset/1000.jsonl 

CUDA_VISIBLE_DEVICES=2,0,1,3 nohup python watermark_reliability_release/generation_pipeline.py --watermark john23 --dataset_name hc3 --run_name gen-hc3-john23 --model_name_or_path meta-llama/Llama-2-7b-chat-hf --min_generations 500 &
CUDA_VISIBLE_DEVICES=1,0,2,3 nohup python watermark_reliability_release/generation_pipeline.py --watermark xuandong23b --dataset_name hc3 --run_name gen-hc3-xuandong23b --model_name_or_path meta-llama/Llama-2-7b-chat-hf --min_generations 500 &

# CUDA_VISIBLE_DEVICES=2,0,1,3 nohup python watermark_reliability_release/generation_pipeline.py --watermark john23 --dataset_name c4 --run_name gen-c4-john23 --model_name_or_path meta-llama/Llama-2-7b-chat-hf --min_generations 500 &
# CUDA_VISIBLE_DEVICES=1,0,2,3 nohup python watermark_reliability_release/generation_pipeline.py --watermark xuandong23b --dataset_name c4 --run_name gen-c4-xuandong23b --model_name_or_path meta-llama/Llama-2-7b-chat-hf --min_generations 500 &