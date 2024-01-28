#!/bin/bash
#===============================================================================
# usecommand: nohup bash aiwei.sh > aiwei23_new_test.log 2>&1 &
#===============================================================================

CUDA_VISIBLE_DEVICES=0,1,2,3 python watermark_reliability_release/generation_pipeline.py --watermark aiwei23 --dataset_name c4 --run_name aiwei_gen_test_llama_50 --model_name_or_path meta-llama/Llama-2-7b-chat-hf --min_generations 50 --max_new_tokens 100 --output_dir aiwei23/c4/llama/test-llama
CUDA_VISIBLE_DEVICES=0,1,2,3 python watermark_reliability_release/generation_pipeline.py --watermark aiwei23 --dataset_name c4 --run_name aiwei_gen_test_opt_50 --model_name_or_path facebook/opt-1.3b --min_generations 50 --max_new_tokens 100 --output_dir aiwei23/c4/opt/test-opt


