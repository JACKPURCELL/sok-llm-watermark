#!/bin/bash
#===============================================================================
# usecommand nohup bash gen.sh > gen.log 2>&1 &
#===============================================================================

# CUDA_VISIBLE_DEVICES=2,0,1,3 python watermark_reliability_release/generation_pipeline.py --watermark kiyoon23 --dataset_name c4 --run_name gen-c4-kiyoon23-llama --output_dir kiyoon23/llama --model_name_or_path meta-llama/Llama-2-7b-chat-hf --min_generations 1000 --max_new_tokens 100
# CUDA_VISIBLE_DEVICES=2,0,1,3 python watermark_reliability_release/generation_pipeline.py --watermark kiyoon23 --dataset_name c4 --run_name gen-c4-kiyoon23-opt --output_dir kiyoon23/opt --model_name_or_path facebook/opt-1.3b --min_generations 1000 --max_new_tokens 100
CUDA_VISIBLE_DEVICES=2,0,1 python watermark_reliability_release/generation_pipeline.py --watermark xiaoniu23 --dataset_name c4 --run_name gen-c4-xiaoniu23-llama --output_dir xiaoniu23/llama --model_name_or_path meta-llama/Llama-2-7b-chat-hf --min_generations 1000 --max_new_tokens 100
CUDA_VISIBLE_DEVICES=2,0,1 python watermark_reliability_release/generation_pipeline.py --watermark xiaoniu23 --dataset_name c4 --run_name gen-c4-xiaoniu23-opt --output_dir xiaoniu23/opt --model_name_or_path facebook/opt-1.3b --min_generations 1000 --max_new_tokens 100