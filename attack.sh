CUDA_VISIBLE_DEVICES=0,1 nohup python watermark_reliability_release/attack_pipeline.py --run_name dipper-attack-xuandong23b --input_dir /home/jkl6486/sok-llm-watermark/runs/xuandong23b/hc3 --output_dir /home/jkl6486/sok-llm-watermark/runs/xuandong23b/hc3/dipper --attack_method dipper & 
CUDA_VISIBLE_DEVICES=1 nohup python watermark_reliability_release/attack_pipeline.py --run_name dipper-attack-john23 --input_dir /home/jkl6486/sok-llm-watermark/runs/john23/hc3 --output_dir /home/jkl6486/sok-llm-watermark/runs/john23/hc3/dipper --attack_method dipper & 
CUDA_VISIBLE_DEVICES=1,0 nohup python watermark_reliability_release/attack_pipeline.py --run_name dipper-attack-lean23 --input_dir /home/jkl6486/sok-llm-watermark/runs/lean23/hc3 --output_dir /home/jkl6486/sok-llm-watermark/runs/lean23/hc3/dipper --attack_method dipper & 
CUDA_VISIBLE_DEVICES=1 nohup python watermark_reliability_release/attack_pipeline.py --run_name dipper-attack-rohith23 --input_dir /home/jkl6486/sok-llm-watermark/runs/rohith23/hc3 --output_dir /home/jkl6486/sok-llm-watermark/runs/rohith23/hc3/dipper --attack_method dipper & 