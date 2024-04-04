    #!/bin/bash
    #===============================================================================
    # usecommand nohup bash shell/finetune_roberta.sh > shell/0402_finetune_roberta_1.log 2>&1 &
    #===============================================================================

    watermark_types=("xuandong23b" "aiwei23" "aiwei23b" "scott22")
    # "john23" "lean23" "rohith23" "xiaoniu23"
    #"john23" "lean23" "rohith23" "xiaoniu23" "xuandong23b" "aiwei23" "aiwei23b"

    gpus=("3")
    dippers=("l40" "l20")

    for watermark_type in "${watermark_types[@]}"; do
    echo "start $watermark_type"
        for dipper in "${dippers[@]}"; do 
            echo "start $dipper"  
            python /home/jkl6486/sok-llm-watermark/watermark_reliability_release/fine_tune_roberta_gpt_data.py --train_path /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/opt/dipper_"$dipper"_o0/gen_table_w_metrics.jsonl --output_path /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/opt/dipper_"$dipper"_o0/dipp_roberta_finetuned --gpus "$gpus" --method_name "$watermark_type"
        done
    done
    echo "finish $watermark_type"
