CUDA_VISIBLE_DEVICES=0,1 nohup python watermark_reliability_release/attack_pipeline.py --run_name dipper-attack-xuandong23b --input_dir /home/jkl6486/sok-llm-watermark/runs/xuandong23b/hc3 --output_dir /home/jkl6486/sok-llm-watermark/runs/xuandong23b/hc3/dipper --attack_method dipper & 
CUDA_VISIBLE_DEVICES=1 nohup python watermark_reliability_release/attack_pipeline.py --run_name dipper-attack-john23 --input_dir /home/jkl6486/sok-llm-watermark/runs/john23/hc3 --output_dir /home/jkl6486/sok-llm-watermark/runs/john23/hc3/dipper --attack_method dipper & 
CUDA_VISIBLE_DEVICES=2,3 nohup python watermark_reliability_release/attack_pipeline.py --run_name dipper-attack-lean23 --input_dir /home/jkl6486/sok-llm-watermark/runs/lean23/hc3 --output_dir /home/jkl6486/sok-llm-watermark/runs/lean23/hc3/dipper --attack_method dipper & 
CUDA_VISIBLE_DEVICES=3 nohup python watermark_reliability_release/attack_pipeline.py --run_name dipper-attack-rohith23 --input_dir /home/jkl6486/sok-llm-watermark/runs/rohith23/hc3 --output_dir /home/jkl6486/sok-llm-watermark/runs/rohith23/hc3/dipper --attack_method dipper & 
CUDA_VISIBLE_DEVICES=2 nohup python watermark_reliability_release/attack_pipeline.py --run_name cp-attack-rohith23 --input_dir /home/jkl6486/sok-llm-watermark/runs/rohith23/hc3  --attack_method copy-paste & 

CUDA_VISIBLE_DEVICES=1 nohup python watermark_reliability_release/attack_pipeline.py --run_name dipper-attack-rohith23-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/rohith23/c4 --output_dir /home/jkl6486/sok-llm-watermark/runs/rohith23/c4/dipper --attack_method dipper & 


CUDA_VISIBLE_DEVICES=3 nohup python watermark_reliability_release/attack_pipeline.py --run_name dipper-attack-aiwei23 --input_dir /home/jkl6486/sok-llm-watermark/runs/aiwei23/hc3 --output_dir /home/jkl6486/sok-llm-watermark/runs/aiwei23/hc3/dipper --attack_method dipper & 
CUDA_VISIBLE_DEVICES=3 nohup python watermark_reliability_release/attack_pipeline.py --run_name cp-attack-aiwei23 --input_dir /home/jkl6486/sok-llm-watermark/runs/aiwei23/hc3  --attack_method copy-paste & 

CUDA_VISIBLE_DEVICES=3 nohup python watermark_reliability_release/attack_pipeline.py --run_name dipper-attack-xuandong23b-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/xuandong23b/c4 --output_dir /home/jkl6486/sok-llm-watermark/runs/xuandong23b/c4/dipper --attack_method dipper & 
CUDA_VISIBLE_DEVICES=2 nohup python watermark_reliability_release/attack_pipeline.py --run_name dipper-attack-john23-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/john23/c4  --output_dir /home/jkl6486/sok-llm-watermark/runs/john23/c4/dipper --attack_method dipper & 
CUDA_VISIBLE_DEVICES=1 nohup python watermark_reliability_release/attack_pipeline.py --run_name dipper-attack-aiwei23-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/aiwei23/c4  --output_dir /home/jkl6486/sok-llm-watermark/runs/aiwei23/c4/dipper --attack_method dipper & 
CUDA_VISIBLE_DEVICES=0 nohup python watermark_reliability_release/attack_pipeline.py --run_name dipper-attack-aiwei23-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/aiwei23/c4  --output_dir /home/jkl6486/sok-llm-watermark/runs/aiwei23/c4 --attack_method copy-paste & 


CUDA_VISIBLE_DEVICES=3 nohup python watermark_reliability_release/attack_pipeline.py --run_name dipper-attack-xuandong23b-lfqa --input_dir /home/jkl6486/sok-llm-watermark/runs/xuandong23b/lfqa/t300 --output_dir /home/jkl6486/sok-llm-watermark/runs/xuandong23b/lfqa/t300/dipper --attack_method dipper & 
CUDA_VISIBLE_DEVICES=2 nohup python watermark_reliability_release/attack_pipeline.py --run_name cp-attack-xuandong23b-lfqa --input_dir /home/jkl6486/sok-llm-watermark/runs/xuandong23b/lfqa/t300  --output_dir /home/jkl6486/sok-llm-watermark/runs/john23/lfqa/t300 --attack_method copy-paste & 

CUDA_VISIBLE_DEVICES=2 nohup python watermark_reliability_release/attack_pipeline.py --run_name dipper-attack-john23-c4-t300 --input_dir /home/jkl6486/sok-llm-watermark/runs/john23/c4/t300 --output_dir /home/jkl6486/sok-llm-watermark/runs/john23/c4/t300/dipper --attack_method dipper & 
CUDA_VISIBLE_DEVICES=2 nohup python watermark_reliability_release/attack_pipeline.py --run_name cp-attack-john23-c4-t300 --input_dir /home/jkl6486/sok-llm-watermark/runs/john23/c4/t300  --output_dir /home/jkl6486/sok-llm-watermark/runs/john23/c4/t300 --attack_method copy-paste & 

CUDA_VISIBLE_DEVICES=1 nohup python watermark_reliability_release/attack_pipeline.py --run_name dipper-attack-rohith23-c4-t300 --input_dir /home/jkl6486/sok-llm-watermark/runs/rohith23/c4/t300 --output_dir /home/jkl6486/sok-llm-watermark/runs/rohith23/c4/t300/dipper --attack_method dipper & 
CUDA_VISIBLE_DEVICES=1 nohup python watermark_reliability_release/attack_pipeline.py --run_name cp-attack-rohith23-c4-t300 --input_dir /home/jkl6486/sok-llm-watermark/runs/rohith23/c4/t300  --output_dir /home/jkl6486/sok-llm-watermark/runs/rohith23/c4/t300 --attack_method copy-paste & 

CUDA_VISIBLE_DEVICES=2 nohup python watermark_reliability_release/attack_pipeline.py --run_name dipper-attack-john23-lfqa --input_dir /home/jkl6486/sok-llm-watermark/runs/john23/lfqa/t300 --output_dir /home/jkl6486/sok-llm-watermark/runs/john23/lfqa/t300/dipper --attack_method dipper & 
CUDA_VISIBLE_DEVICES=2 nohup python watermark_reliability_release/attack_pipeline.py --run_name cp-attack-john23-lfqa --input_dir /home/jkl6486/sok-llm-watermark/runs/john23/lfqa/t300  --output_dir /home/jkl6486/sok-llm-watermark/runs/john23/lfqa/t300 --attack_method copy-paste & 

CUDA_VISIBLE_DEVICES=1 nohup python watermark_reliability_release/attack_pipeline.py --run_name dipper-attack-rohith23-lfqa --input_dir /home/jkl6486/sok-llm-watermark/runs/rohith23/lfqa/t300 --output_dir /home/jkl6486/sok-llm-watermark/runs/rohith23/lfqa/t300/dipper --attack_method dipper & 
CUDA_VISIBLE_DEVICES=1 nohup python watermark_reliability_release/attack_pipeline.py --run_name cp-attack-rohith23-lfqa --input_dir /home/jkl6486/sok-llm-watermark/runs/rohith23/lfqa/t300  --output_dir /home/jkl6486/sok-llm-watermark/runs/rohith23/lfqa/t300 --attack_method copy-paste & 


CUDA_VISIBLE_DEVICES=0 nohup python watermark_reliability_release/attack_pipeline.py --run_name dipper-attack-xuandong23b-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/xuandong23b/c4/t300 --output_dir /home/jkl6486/sok-llm-watermark/runs/xuandong23b/c4/t300/dipper --attack_method dipper & 
CUDA_VISIBLE_DEVICES=1 nohup python watermark_reliability_release/attack_pipeline.py --run_name cp-attack-xuandong23b-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/xuandong23b/c4/t300  --output_dir /home/jkl6486/sok-llm-watermark/runs/xuandong23b/c4/t300 --attack_method copy-paste & 

CUDA_VISIBLE_DEVICES=0 nohup python watermark_reliability_release/attack_pipeline.py --run_name dipper-attack-xuandong23b-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/xuandong23b/c4/t300 --output_dir /home/jkl6486/sok-llm-watermark/runs/xuandong23b/c4/t300/dipper --attack_method dipper & 
CUDA_VISIBLE_DEVICES=1 nohup python watermark_reliability_release/attack_pipeline.py --run_name cp-attack-xuandong23b-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/xuandong23b/c4/t300  --output_dir /home/jkl6486/sok-llm-watermark/runs/xuandong23b/c4/t300 --attack_method copy-paste & 

CUDA_VISIBLE_DEVICES=2 nohup python watermark_reliability_release/attack_pipeline.py --run_name dipper-attack-xuandong23b-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/xuandong23b/c4/t200 --output_dir /home/jkl6486/sok-llm-watermark/runs/xuandong23b/c4/t200/dipper --attack_method dipper & 
CUDA_VISIBLE_DEVICES=1 nohup python watermark_reliability_release/attack_pipeline.py --run_name cp-attack-xuandong23b-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/xuandong23b/c4/t200  --output_dir /home/jkl6486/sok-llm-watermark/runs/xuandong23b/c4/t200 --attack_method copy-paste &