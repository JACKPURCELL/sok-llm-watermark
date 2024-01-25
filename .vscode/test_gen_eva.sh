#!/bin/bash
#===============================================================================
# usecommand: nohup bash test_gen_eva.sh > test_gen_eva.log 2>&1 &
#===============================================================================
watermark_types=("xuandong23b" "xiaoniu23" "rohith23" "kiyoon23" "lean23")

# "xuandong23b" "rohith23" "lean23" "kiyoon23" "xiaoniu23"

for watermark_type in "${watermark_types[@]}"; do
    echo "start $watermark_type generation"

        CUDA_VISIBLE_DEVICES=0,1,2,3 python watermark_reliability_release/generation_pipeline.py --watermark "$watermark_type" --dataset_name c4 --run_name "$watermark_type"-gen-test --model_name_or_path facebook/opt-1.3b --min_generations 50 --max_new_tokens 100 --output_dir runs_test/"$watermark_type"  
        
     echo "finish $watermark_type generation"

     echo "start $watermark_type evaluation"

        CUDA_VISIBLE_DEVICES=0,1,2,3 python watermark_reliability_release/evaluation_pipeline.py --watermark "$watermark_type" --run_name "$watermark_type"-eva-test --input_dir runs_test/"$watermark_type" --evaluation_metrics z-score 

    echo "finish $watermark_type evaluation"

done
echo "finish all"