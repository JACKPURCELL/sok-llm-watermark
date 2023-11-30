#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python main.py --watermark xuandong23b --prompt_file /home/jkl6486/sok-llm-watermark/dataset/1000.jsonl > runs/xuandong23b.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 python main.py --watermark lean23 --prompt_file /home/jkl6486/sok-llm-watermark/dataset/1000.jsonl > runs/lean23.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 python main.py --watermark john23 --prompt_file /home/jkl6486/sok-llm-watermark/dataset/1000.jsonl > runs/john23.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python main.py --watermark xuandong23b --attack dipper --skip_inject True --prompt_file /home/jkl6486/sok-llm-watermark/dataset/1000.jsonl > runs/xuandong23b_attack.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 python main.py --watermark lean23 --attack dipper --skip_inject True --prompt_file /home/jkl6486/sok-llm-watermark/dataset/1000.jsonl > runs/lean23_attack.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 python main.py --watermark john23 --attack dipper --skip_inject True --prompt_file /home/jkl6486/sok-llm-watermark/dataset/1000.jsonl > runs/john23_attack.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=1  python main.py --watermark rohith23  --prompt_file /home/jkl6486/sok-llm-watermark/dataset/1000.jsonl 

