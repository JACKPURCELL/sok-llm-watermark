    #!/bin/bash
    #===============================================================================
    # usecommand nohup bash shell/finetune_roberta.sh > shell/finetune_roberta.log 2>&1 &
    #===============================================================================

    watermark_types=("john23" "lean23" "rohith23" "xiaoniu23" "xuandong23b" "aiwei23" "aiwei23b")
    #"john23" "lean23" "rohith23" "xiaoniu23" "xuandong23b" "aiwei23" "aiwei23b"

    gpus=("0, 3")
    dippers=("l40" "l20")

    for watermark_type in "${watermark_types[@]}"; do
    echo "start $watermark_type"
        for dipper in "${dippers[@]}"; do 
            echo "start $dipper"  
            python /home/jkl6486/sok-llm-watermark/watermark_reliability_release/fine_tune_roberta.py --train_path /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/llama/gen_table.jsonl --output_path /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/llama/roberta_finetuned_machinevsmachine
        done
    done
    echo "finish $watermark_type"
