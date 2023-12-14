CUDA_VISIBLE_DEVICES=1,2 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --overwrite_output_file True --watermark xuandong23b --run_name eva-xuandong23b-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/xuandong23b/c4 --evaluation_metrics all &
CUDA_VISIBLE_DEVICES=2,1 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --overwrite_output_file True --watermark xuandong23b --run_name eva-xuandong23b-hc3 --input_dir /home/jkl6486/sok-llm-watermark/runs/xuandong23b/hc3 --evaluation_metrics all &
CUDA_VISIBLE_DEVICES=3,2 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --overwrite_output_file True --watermark john23 --run_name eva-john23-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/john23/c4 --evaluation_metrics all &
CUDA_VISIBLE_DEVICES=3,2 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --overwrite_output_file True --watermark john23 --run_name eva-john23-hc3 --input_dir /home/jkl6486/sok-llm-watermark/runs/john23/hc3 --evaluation_metrics all &
CUDA_VISIBLE_DEVICES=3,2 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --overwrite_output_file True --watermark lean23 --run_name eva-lean23-hc3 --input_dir /home/jkl6486/sok-llm-watermark/runs/lean23/hc3 --evaluation_metrics all &




CUDA_VISIBLE_DEVICES=2,1,3,0 watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark xuandong23b --run_name eva-rohith23-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/rohith23/c4 --evaluation_metrics all 
CUDA_VISIBLE_DEVICES=3,0,1,2 watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark xuandong23b --run_name eva-rohith23-hc3 --input_dir /home/jkl6486/sok-llm-watermark/runs/rohith23/hc3 --evaluation_metrics all 