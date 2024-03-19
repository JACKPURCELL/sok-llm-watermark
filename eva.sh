CUDA_VISIBLE_DEVICES=1,2 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --overwrite_output_file True --watermark xuandong23b --run_name eva-xuandong23b-c4 --input_dir ~/sok-llm-watermark/runs/xuandong23b/c4 --evaluation_metrics all &
CUDA_VISIBLE_DEVICES=2,1 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --overwrite_output_file True --watermark xuandong23b --run_name eva-xuandong23b-hc3 --input_dir ~/sok-llm-watermark/runs/xuandong23b/hc3 --evaluation_metrics all &
CUDA_VISIBLE_DEVICES=3,2 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --overwrite_output_file True --watermark john23 --run_name eva-john23-c4 --input_dir ~/sok-llm-watermark/runs/john23/c4 --evaluation_metrics all &
CUDA_VISIBLE_DEVICES=3,2 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --overwrite_output_file True --watermark john23 --run_name eva-john23-hc3 --input_dir ~/sok-llm-watermark/runs/john23/hc3 --evaluation_metrics all &
CUDA_VISIBLE_DEVICES=3,2 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --overwrite_output_file True --watermark lean23 --run_name eva-lean23-hc3 --input_dir ~/sok-llm-watermark/runs/lean23/hc3 --evaluation_metrics all &


CUDA_VISIBLE_DEVICES=0 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark john23 --run_name eva-john23-c4-dipper --input_dir ~/sok-llm-watermark/runs/john23/c4/dipper --evaluation_metrics all &
CUDA_VISIBLE_DEVICES=1 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark aiwei23 --run_name eva-aiwei23-c4-dipper --input_dir ~/sok-llm-watermark/runs/aiwei23/c4/dipper --evaluation_metrics all &
CUDA_VISIBLE_DEVICES=1 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark aiwei23 --run_name eva-aiwei23-c4 --input_dir ~/sok-llm-watermark/runs/aiwei23/c4 --evaluation_metrics all &
CUDA_VISIBLE_DEVICES=2 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark xuandong23b --run_name eva-xuandong23b-c4-dipper --input_dir ~/sok-llm-watermark/runs/xuandong23b/c4/dipper --evaluation_metrics all &

CUDA_VISIBLE_DEVICES=0,1 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark aiwei23 --run_name eva-aiwei23-hc3 --input_dir ~/sok-llm-watermark/runs/aiwei23/hc3 --evaluation_metrics all &
CUDA_VISIBLE_DEVICES=1,0 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark aiwei23 --run_name eva-aiwei23-hc3-dipper --input_dir ~/sok-llm-watermark/runs/aiwei23/hc3/dipper --evaluation_metrics all &

CUDA_VISIBLE_DEVICES=1,3 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark rohith23 --run_name eva-rohith23-c4 --input_dir ~/sok-llm-watermark/runs/rohith23/c4 --evaluation_metrics all &
nCUDA_VISIBLE_DEVICES=0,1 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark rohith23 --run_name eva-rohith23-hc3 --input_dir ~/sok-llm-watermark/runs/rohith23/hc3 --evaluation_metrics all &
CUDA_VISIBLE_DEVICES=1,2 ohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark rohith23 --run_name eva-rohith23-hc3-dipper --input_dir ~/sok-llm-watermark/runs/rohith23/hc3/dipper --evaluation_metrics all &
CUDA_VISIBLE_DEVICES=1,3  python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark rohith23 --run_name eva-rohith23-hc3 --input_dir ~/sok-llm-watermark/runs/rohith23/hc3 --evaluation_metrics all 
CUDA_VISIBLE_DEVICES=3,2 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark john23 --run_name eva-john23-hc3 --input_dir ~/sok-llm-watermark/runs/john23/hc3/dipper --evaluation_metrics all &
CUDA_VISIBLE_DEVICES=3,2 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark xuandong23b --run_name eva-xuandong23b-hc3 --input_dir ~/sok-llm-watermark/runs/xuandong23b/hc3/dipper --evaluation_metrics all &

CUDA_VISIBLE_DEVICES=0 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark xuandong23b --run_name eva-xuandong23b-c4-t300-dipper --input_dir ~/sok-llm-watermark/runs/xuandong23b/c4/t300/dipper --evaluation_metrics all &
CUDA_VISIBLE_DEVICES=1 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark xuandong23b --run_name eva-xuandong23b-c4-t300 --input_dir ~/sok-llm-watermark/runs/xuandong23b/c4/t300 --evaluation_metrics all &


CUDA_VISIBLE_DEVICES=0 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark xuandong23b --run_name eva-xuandong23b-lfqa-t300-dipper --input_dir ~/sok-llm-watermark/runs/xuandong23b/lfqa/t300/dipper --evaluation_metrics all &
CUDA_VISIBLE_DEVICES=0 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark xuandong23b --run_name eva-xuandong23b-lfqa-t300 --input_dir ~/sok-llm-watermark/runs/xuandong23b/lfqa/t300 --evaluation_metrics all &

CUDA_VISIBLE_DEVICES=2 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark john23 --run_name eva-john23-c4-t300-dipper --input_dir ~/sok-llm-watermark/runs/john23/c4/t300/dipper --evaluation_metrics all &
CUDA_VISIBLE_DEVICES=2 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark john23 --run_name eva-john23-c4-t300 --input_dir ~/sok-llm-watermark/runs/john23/c4/t300 --evaluation_metrics all &

CUDA_VISIBLE_DEVICES=2 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark rohith23 --run_name eva-rohith23-c4-t300-dipper --input_dir ~/sok-llm-watermark/runs/rohith23/c4/t300/dipper --evaluation_metrics all &
CUDA_VISIBLE_DEVICES=2 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark rohith23 --run_name eva-rohith23-c4-t300 --input_dir ~/sok-llm-watermark/runs/rohith23/c4/t300 --evaluation_metrics all &


CUDA_VISIBLE_DEVICES=1 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark john23 --run_name eva-john23-lfqa-t300-dipper --input_dir ~/sok-llm-watermark/runs/john23/lfqa/t300/dipper --evaluation_metrics all &
CUDA_VISIBLE_DEVICES=1 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark john23 --run_name eva-john23-lfqa-t300 --input_dir ~/sok-llm-watermark/runs/john23/lfqa/t300 --evaluation_metrics all &


CUDA_VISIBLE_DEVICES=1 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark john23 --run_name eva-john23-c4-t200-dipper --input_dir ~/sok-llm-watermark/runs/john23/c4/t200/dipper --evaluation_metrics all &
CUDA_VISIBLE_DEVICES=1 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark john23 --run_name eva-john23-c4-t200 --input_dir ~/sok-llm-watermark/runs/john23/c4/t200 --evaluation_metrics all &


CUDA_VISIBLE_DEVICES=0 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark john23 --run_name eva-john23-c4-t100 --input_dir ~/sok-llm-watermark/runs/john23/c4 --evaluation_metrics all --overwrite_output_file True --lower_tolerance_T 50 --upper_tolerance_T 50 & 
CUDA_VISIBLE_DEVICES=1 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark john23 --run_name eva-john23-c4-t200 --input_dir ~/sok-llm-watermark/runs/john23/c4/t200 --evaluation_metrics all --overwrite_output_file True --lower_tolerance_T 150 --upper_tolerance_T 150 &
CUDA_VISIBLE_DEVICES=3 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark john23 --run_name eva-john23-c4-t300 --input_dir ~/sok-llm-watermark/runs/john23/c4/t300 --evaluation_metrics all --overwrite_output_file True --lower_tolerance_T 200 --upper_tolerance_T 200 &


CUDA_VISIBLE_DEVICES=0 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark john23 --run_name eva-john23-c4-t100-dipper --input_dir ~/sok-llm-watermark/runs/john23/c4/dipper --evaluation_metrics all --overwrite_output_file True --lower_tolerance_T 50 --upper_tolerance_T 50 & 
CUDA_VISIBLE_DEVICES=1 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark john23 --run_name eva-john23-c4-t200-dipper --input_dir ~/sok-llm-watermark/runs/john23/c4/t200/dipper --evaluation_metrics all --overwrite_output_file True --lower_tolerance_T 150 --upper_tolerance_T 150 &
CUDA_VISIBLE_DEVICES=3 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark john23 --run_name eva-john23-c4-t300-dipper --input_dir ~/sok-llm-watermark/runs/john23/c4/t300/dipper --evaluation_metrics all --overwrite_output_file True --lower_tolerance_T 200 --upper_tolerance_T 200 &
CUDA_VISIBLE_DEVICES=3 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark john23 --run_name eva-john23-lfqa-t300-dipper --input_dir ~/sok-llm-watermark/runs/john23/lfqa/t300/dipper --evaluation_metrics all --overwrite_output_file True --lower_tolerance_T 200 --upper_tolerance_T 200 &
CUDA_VISIBLE_DEVICES=3 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark john23 --run_name eva-john23-lfqa-t300 --input_dir ~/sok-llm-watermark/runs/john23/lfqa/t300 --evaluation_metrics all --overwrite_output_file True --lower_tolerance_T 200 --upper_tolerance_T 200 &


CUDA_VISIBLE_DEVICES=1 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark xuandong23b --run_name eva-xuandong23b-c4-t100 --input_dir ~/sok-llm-watermark/runs/xuandong23b/c4 --evaluation_metrics all --overwrite_output_file True --lower_tolerance_T 50 --upper_tolerance_T 50 & 
CUDA_VISIBLE_DEVICES=2 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark xuandong23b --run_name eva-xuandong23b-c4-t200 --input_dir ~/sok-llm-watermark/runs/xuandong23b/c4/t200 --evaluation_metrics all --overwrite_output_file True --lower_tolerance_T 150 --upper_tolerance_T 150 &
CUDA_VISIBLE_DEVICES=0 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark xuandong23b --run_name eva-xuandong23b-c4-t300 --input_dir ~/sok-llm-watermark/runs/xuandong23b/c4/t300 --evaluation_metrics all --overwrite_output_file True --lower_tolerance_T 200 --upper_tolerance_T 200 &
CUDA_VISIBLE_DEVICES=0 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark xuandong23b --run_name eva-xuandong23b-lfqa-t300 --input_dir ~/sok-llm-watermark/runs/xuandong23b/lfqa/t300 --evaluation_metrics all --overwrite_output_file True --lower_tolerance_T 200 --upper_tolerance_T 200 &
CUDA_VISIBLE_DEVICES=1 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark xuandong23b --run_name eva-xuandong23b-lfqa-t300-dipper --input_dir ~/sok-llm-watermark/runs/xuandong23b/lfqa/t300/dipper --evaluation_metrics all --overwrite_output_file True --lower_tolerance_T 200 --upper_tolerance_T 200 &


CUDA_VISIBLE_DEVICES=0 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark xuandong23b --run_name eva-xuandong23b-c4-t100-dipper --input_dir ~/sok-llm-watermark/runs/xuandong23b/c4/dipper --evaluation_metrics all --overwrite_output_file True --lower_tolerance_T 50 --upper_tolerance_T 50 & 
CUDA_VISIBLE_DEVICES=0 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark xuandong23b --run_name eva-xuandong23b-c4-t200-dipper --input_dir ~/sok-llm-watermark/runs/xuandong23b/c4/t200/dipper --evaluation_metrics all --overwrite_output_file True --lower_tolerance_T 150 --upper_tolerance_T 150 &
CUDA_VISIBLE_DEVICES=3 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark xuandong23b --run_name eva-xuandong23b-c4-t300-dipper --input_dir ~/sok-llm-watermark/runs/xuandong23b/c4/t300/dipper --evaluation_metrics all --overwrite_output_file True --lower_tolerance_T 200 --upper_tolerance_T 200 &