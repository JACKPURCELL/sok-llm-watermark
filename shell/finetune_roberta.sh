#!/bin/bash
    #===============================================================================
    # usecommand nohup bash shell/finetune_roberta.sh > shell/0421_finetune_roberta_chatgpt.log 2>&1 &
    #===============================================================================
    # watermark_types=("john23" "scott22" "rohith23" "xuandong23b")
    watermark_types=("scott22")
    # "aiwei23" "aiwei23b" "xiaoniu23" "scott22"
    # "xuandong23b" "aiwei23" "aiwei23b" "scott22"
    #"john23" "lean23" "rohith23" "xiaoniu23"
    # "xuandong23b" "aiwei23" "aiwei23b" "scott22"
    #"john23" "lean23" "rohith23" "xiaoniu23" "xuandong23b" "aiwei23" "aiwei23b" "scott22"
    gpus=("2")
    dippers=("l40")
    for watermark_type in "${watermark_types[@]}"; do
    echo "start $watermark_type"
        for dipper in "${dippers[@]}"; do
            echo "start $dipper"
            CUDA_VISIBLE_DEVICES="$gpus" python /home/jkl6486/sok-llm-watermark/generic_detector/fine_tune_roberta_gpt_data.py --train_path /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/opt/dipper_"$dipper"_o0/gen_table_w_metrics.jsonl  --output_path /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/opt/dipper_"$dipper"_o0/dipp_roberta_finetuned_chatgpt_new  --method_name "$watermark_type"
        done
    done
    echo "finish $watermark_type"