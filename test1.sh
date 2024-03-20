#!/bin/bash
# CUDA_VISIBLE_DEVICES=1 python main.py --watermark xuandong23b --prompt_file /home/ljc/sok-llm-watermark/dataset/1000.jsonl > runs/xuandong23b.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python main.py --watermark lean23 --prompt_file /home/ljc/sok-llm-watermark/dataset/1000.jsonl > runs/lean23.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python main.py --watermark john23 --prompt_file /home/ljc/sok-llm-watermark/dataset/1000.jsonl > runs/john23.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=1 python main.py --watermark xuandong23b --attack dipper --skip_inject True --prompt_file /home/ljc/sok-llm-watermark/dataset/1000.jsonl > runs/xuandong23b_attack.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1 python main.py --watermark lean23 --attack dipper --skip_inject True --prompt_file /home/ljc/sok-llm-watermark/dataset/1000.jsonl > runs/lean23_attack.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python main.py --watermark john23 --attack dipper --skip_inject True --prompt_file /home/ljc/sok-llm-watermark/dataset/1000.jsonl > runs/john23_attack.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=1  python main.py --watermark rohith23  --prompt_file /home/ljc/sok-llm-watermark/dataset/1000.jsonl 

CUDA_VISIBLE_DEVICES=1,2,3 nohup python watermark_reliability_release/generation_pipeline.py --watermark john23 --dataset_name hc3 --run_name gen-hc3-john23 --model_name_or_path meta-llama/Llama-2-7b-chat-hf --min_generations 1000 &
CUDA_VISIBLE_DEVICES=1,2,3 nohup python watermark_reliability_release/generation_pipeline.py --watermark xuandong23b --dataset_name hc3 --run_name gen-hc3-xuandong23b --model_name_or_path meta-llama/Llama-2-7b-chat-hf --min_generations 1000 &
CUDA_VISIBLE_DEVICES=0,1,2 nohup python watermark_reliability_release/generation_pipeline.py --watermark rohith23 --dataset_name hc3 --run_name gen-hc3-rohith23 --model_name_or_path meta-llama/Llama-2-7b-chat-hf --min_generations 1000 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python watermark_reliability_release/generation_pipeline.py --watermark lean23 --dataset_name hc3 --run_name gen-hc3-lean23 --model_name_or_path meta-llama/Llama-2-7b-chat-hf --min_generations 1000 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python watermark_reliability_release/generation_pipeline.py --watermark kiyoon23 --dataset_name hc3 --run_name gen-hc3-kiyoon23 --model_name_or_path meta-llama/Llama-2-7b-chat-hf --min_generations 1000 &
CUDA_VISIBLE_DEVICES=3,2,1 nohup python watermark_reliability_release/generation_pipeline.py --watermark aiwei23 --dataset_name hc3 --run_name gen-hc3-aiwei23 --model_name_or_path meta-llama/Llama-2-7b-chat-hf --min_generations 1000 &


CUDA_VISIBLE_DEVICES=1,2,3 nohup python watermark_reliability_release/generation_pipeline.py --watermark john23 --dataset_name c4 --run_name gen-c4-john23 --model_name_or_path meta-llama/Llama-2-7b-chat-hf --min_generations 1000 &
CUDA_VISIBLE_DEVICES=1,2,3 nohup python watermark_reliability_release/generation_pipeline.py --watermark xuandong23b --dataset_name c4 --run_name gen-c4-xuandong23b --model_name_or_path meta-llama/Llama-2-7b-chat-hf --min_generations 1000 &
CUDA_VISIBLE_DEVICES=1,2,3 nohup python watermark_reliability_release/generation_pipeline.py --watermark rohith23 --dataset_name c4 --run_name gen-c4-rohith23 --model_name_or_path meta-llama/Llama-2-7b-chat-hf --min_generations 1000 &
CUDA_VISIBLE_DEVICES=3,2,1 nohup python watermark_reliability_release/generation_pipeline.py --watermark lean23 --dataset_name c4 --run_name gen-c4-lean23 --model_name_or_path meta-llama/Llama-2-7b-chat-hf --min_generations 1000 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python watermark_reliability_release/generation_pipeline.py --watermark kiyoon23 --dataset_name c4 --run_name gen-c4-kiyoon23 --model_name_or_path meta-llama/Llama-2-7b-chat-hf --min_generations 1000 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python watermark_reliability_release/generation_pipeline.py --watermark aiwei23 --dataset_name c4 --run_name gen-c4-aiwei23 --model_name_or_path meta-llama/Llama-2-7b-chat-hf --min_generations 1000 &



CUDA_VISIBLE_DEVICES=3,2,1 nohup python watermark_reliability_release/generation_pipeline.py --watermark aiwei23 --dataset_name hc3 --run_name test-gen-aiwei23 --model_name_or_path meta-llama/Llama-2-7b-chat-hf --min_generations 1000 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python watermark_reliability_release/generation_pipeline.py --watermark aiwei23 --dataset_name c4 --run_name gen-c4-aiwei23 --model_name_or_path meta-llama/Llama-2-7b-chat-hf --min_generations 100 &
# CUDA_VISIBLE_DEVICES=2,0,1,3 nohup python watermark_reliability_release/generation_pipeline.py --watermark john23 --dataset_name c4 --run_name gen-c4-john23 --model_name_or_path meta-llama/Llama-2-7b-chat-hf --min_generations 500 &
# CUDA_VISIBLE_DEVICES=1,0,2,3 nohup python watermark_reliability_release/generation_pipeline.py --watermark xuandong23b --dataset_name c4 --run_name gen-c4-xuandong23b --model_name_or_path meta-llama/Llama-2-7b-chat-hf --min_generations 500 &

#token300
CUDA_VISIBLE_DEVICES=1,2 nohup python watermark_reliability_release/generation_pipeline.py --watermark john23 --dataset_name c4 --run_name gen-c4-john23-t300 --max_new_tokens 300 --model_name_or_path meta-llama/Llama-2-7b-chat-hf --output_dir john23/c4/t300 --min_generations 1000 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python watermark_reliability_release/generation_pipeline.py --watermark xuandong23b --dataset_name c4 --run_name gen-c4-xuandong23b-t300 --max_new_tokens 300 --model_name_or_path meta-llama/Llama-2-7b-chat-hf --output_dir xuandong23b/c4/t300 --min_generations 1000 &

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python watermark_reliability_release/generation_pipeline.py --watermark john23 --dataset_name lfqa --run_name gen-lfqa-john23-t300 --max_new_tokens 300 --model_name_or_path meta-llama/Llama-2-7b-chat-hf --output_dir john23/lfqa/t300 --min_generations 1000 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python watermark_reliability_release/generation_pipeline.py --watermark xuandong23b --dataset_name lfqa --run_name gen-lfqa-xuandong23b-t300 --max_new_tokens 300 --model_name_or_path meta-llama/Llama-2-7b-chat-hf --output_dir xuandong23b/lfqa/t300 --min_generations 1000 &

CUDA_VISIBLE_DEVICES=1,2,3 nohup python watermark_reliability_release/generation_pipeline.py --watermark rohith23 --dataset_name c4 --run_name gen-c4-rohith23-t300 --max_new_tokens 300 --model_name_or_path meta-llama/Llama-2-7b-chat-hf --output_dir rohith23/c4/t300 --min_generations 1000 &
CUDA_VISIBLE_DEVICES=3,2,1 nohup python watermark_reliability_release/generation_pipeline.py --watermark lean23 --dataset_name c4 --run_name gen-c4-lean23-t300 --max_new_tokens 300 --model_name_or_path meta-llama/Llama-2-7b-chat-hf --output_dir lean23/c4/t300 --min_generations 1000 &


# CUDA_VISIBLE_DEVICES=0,1 nohup python watermark_reliability_release/detectgpt/detectgpt_main.py --data_path runs/john23/c4/dipper/gen_table_attacked.jsonl &
# CUDA_VISIBLE_DEVICES=3,1 nohup python watermark_reliability_release/detectgpt/detectgpt_main.py --data_path runs/xuandong23b/c4/dipper/gen_table_attacked.jsonl &