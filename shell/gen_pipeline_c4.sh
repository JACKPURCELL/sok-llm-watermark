# !/bin/bash
# ===============================================================================
# usecommand nohup bash shell/gen_pipeline_c4.sh > shell/0128_aiwei_new.log 2>&1 &
# ===============================================================================
echo "Current date and time: $(date)"

watermark_types=("aiwei23" )
# watermark_types=( "rohith23" "xiaoniu23" "xuandong23b" "lean23"  "aiwei23" "aiwei23b")

models=("qwen")
gpus=("0,6,7")

for watermark_type in "${watermark_types[@]}"; do
echo "start $watermark_type"
    for model in "${models[@]}"; do
    echo "start $model"s
        
        CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/generation_pipeline.py --overwrite True --watermark "$watermark_type" --wandb_project qwen --dataset_name arxiv --run_name gen-arxiv-"$watermark_type"-"$model"  --output_dir token_200/"$watermark_type"/arxiv/"$model" --min_generations 50 --max_new_tokens 200 --use_sampling False --generation_batch_size 1 --num_beams 1 --model_name_or_path Qwen/Qwen2.5-14B-Instruct-1M --input_truncation_strategy no_truncation 
        CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --overwrite_output_file True --watermark "$watermark_type" --wandb_project arxiv --run_name eva-c4-"$watermark_type"-"$model"  --input_dir /data/jiacheng/sok-llm-watermark/runs/token_200/"$watermark_type"/arxiv/"$model" --overwrite_args True --evaluation_metrics z-score
    echo "finish $model"

    done
echo "finish $watermark_type"
done
echo "Current date and time: $(date)"


watermark_types=( "aiwei23")
# ===============================================================================
attack_types=("swap" "translation")

synonym_probs=( "0.4" )
# helm_attack_methods=("MisspellingAttack" "TypoAttack" )
helm_attack_methods=("MisspellingAttack" "TypoAttack" "ContractionAttack" "LowercaseAttack" "ExpansionAttack")
dipper_lexs=("20" "40")
cp_attack_insertion_lens=("10" "25")

gpus=("0,6,7")




#====
models=("qwen")
echo "Current date and time: $(date)"
# 遍历数组中的每个元素
for watermark_type in "${watermark_types[@]}"; do
    echo "start $watermark_type"
    for model in "${models[@]}"; do
            # dipper need 48gb memory


        # # # #synonym
         for synonym_prob in "${synonym_probs[@]}"; do
             CUDA_VISIBLE_DEVICES="$gpus" python  watermark_reliability_release/attack_pipeline.py --overwrite_args True --overwrite_output_file True --wandb_project arxiv-attack --run_name synonym-"$synonym_prob"-attack-"$watermark_type"-arxiv --input_dir /data/jiacheng/sok-llm-watermark/runs/token_200/"$watermark_type"/arxiv/"$model"  --output_dir /data/jiacheng/sok-llm-watermark/runs/token_200/"$watermark_type"/arxiv/"$model"/synonym-"$synonym_prob" --attack_method synonym --synonym_p "$synonym_prob"
         done

        # # # copypaste
        for cp_attack_insertion_len in "${cp_attack_insertion_lens[@]}"; do
            CUDA_VISIBLE_DEVICES="$gpus" python  watermark_reliability_release/attack_pipeline.py --cp_attack_type single-single --overwrite_args True --overwrite_output_file True --wandb_project arxiv-attack --run_name copypaste-1-"$cp_attack_insertion_len"-attack-"$watermark_type"-arxiv --input_dir /data/jiacheng/sok-llm-watermark/runs/token_200/"$watermark_type"/arxiv/"$model"  --output_dir /data/jiacheng/sok-llm-watermark/runs/token_200/"$watermark_type"/arxiv/"$model"/copypaste-1-"$cp_attack_insertion_len" --attack_method copy-paste --cp_attack_insertion_len "$cp_attack_insertion_len"
            # if [ $cp_attack_insertion_len != "25" ]; then
            CUDA_VISIBLE_DEVICES="$gpus" python  watermark_reliability_release/attack_pipeline.py --cp_attack_type triple-single --overwrite_args True --overwrite_output_file True --wandb_project arxiv-attack --run_name copypaste-3-"$cp_attack_insertion_len"-attack-"$watermark_type"-arxiv --input_dir /data/jiacheng/sok-llm-watermark/runs/token_200/"$watermark_type"/arxiv/"$model"  --output_dir /data/jiacheng/sok-llm-watermark/runs/token_200/"$watermark_type"/arxiv/"$model"/copypaste-3-"$cp_attack_insertion_len" --attack_method copy-paste --cp_attack_insertion_len "$cp_attack_insertion_len"
            # fi
        done

        # #helm
        for helm_attack_method in "${helm_attack_methods[@]}"; do
            CUDA_VISIBLE_DEVICES="$gpus" python  watermark_reliability_release/attack_pipeline.py --overwrite_args True --overwrite_output_file True --wandb_project arxiv-attack --run_name helm-"$helm_attack_method"-attack-"$watermark_type"-arxiv --input_dir /data/jiacheng/sok-llm-watermark/runs/token_200/"$watermark_type"/arxiv/"$model"  --output_dir /data/jiacheng/sok-llm-watermark/runs/token_200/"$watermark_type"/arxiv/"$model"/"$helm_attack_method" --attack_method helm --helm_attack_method "$helm_attack_method"
        done



         for attack_type in "${attack_types[@]}"; do
        #      # 对每个元素执行CUDA_VISIBLE_DEVICES="$gpus" python 命令
             # CUDA_VISIBLE_DEVICES="$gpus" python  main.py --watermark "$attack_type"
             CUDA_VISIBLE_DEVICES="$gpus" python  watermark_reliability_release/attack_pipeline.py --overwrite_args True --overwrite_output_file True --wandb_project arxiv-attack --run_name "$attack_type"-attack-"$watermark_type"-arxiv --input_dir /data/jiacheng/sok-llm-watermark/runs/token_200/"$watermark_type"/arxiv/"$model"  --output_dir /data/jiacheng/sok-llm-watermark/runs/token_200/"$watermark_type"/arxiv/"$model"/"$attack_type" --attack_method "$attack_type"
         done


        for dipper_lex in "${dipper_lexs[@]}"; do
        echo "Current date and time: $(date)"

            CUDA_VISIBLE_DEVICES="$gpus" python  watermark_reliability_release/attack_pipeline.py --overwrite_args True --overwrite_output_file True --wandb_project arxiv-attack --run_name dipper-attack-"$watermark_type"-arxiv --input_dir /data/jiacheng/sok-llm-watermark/runs/token_200/"$watermark_type"/arxiv/"$model"  --output_dir /data/jiacheng/sok-llm-watermark/runs/token_200/"$watermark_type"/arxiv/"$model"/dipper_l"$dipper_lex"_o0 --attack_method dipper --lex "$dipper_lex" 
        done
        # CUDA_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES="$gpus" python  watermark_reliability_release/attack_pipeline.py --overwrite_args True --overwrite_output_file True --wandb_project arxiv-attack --run_name dipper-attack-"$watermark_type"-arxiv --input_dir /data/jiacheng/sok-llm-watermark/runs/token_200/"$watermark_type"/arxiv/"$model"  --output_dir /data/jiacheng/sok-llm-watermark/runs/token_200/"$watermark_type"/arxiv/"$model"/dipper_l60_o20 --attack_method dipper --lex 60 --order 20
        echo "finish $model"
    done
    echo "finish $watermark_type"
done
echo "finish all"
echo "Current date and time: $(date)"





tokens=( "token_200")
#====

for watermark_type in "${watermark_types[@]}"; do
echo "start $watermark_type"
    for model in "${models[@]}"; do
    echo "start $model"
        for token in  "${tokens[@]}"; do
    echo "start $token"


        # # #synonym
        for synonym_prob in "${synonym_probs[@]}"; do
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True   --overwrite_output_file True --watermark "$watermark_type" --run_name synonym-"$synonym_prob"-neweva-"$watermark_type"-arxiv-"$model"  --input_dir /data/jiacheng/sok-llm-watermark/runs/"$token"/"$watermark_type"/arxiv/"$model"/synonym-"$synonym_prob" --wandb_project eva-aiwei23b-attack --overwrite_args True --evaluation_metrics z-score
        done

        # # # copypaste
        for cp_attack_insertion_len in "${cp_attack_insertion_lens[@]}"; do
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True --overwrite_output_file True --watermark "$watermark_type" --run_name copypaste-1-"$cp_attack_insertion_len"-neweva-"$watermark_type"-arxiv-"$model"  --input_dir /data/jiacheng/sok-llm-watermark/runs/"$token"/"$watermark_type"/arxiv/"$model"/copypaste-1-"$cp_attack_insertion_len"  --wandb_project temporary --overwrite_args True --evaluation_metrics z-score
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True --overwrite_output_file True --watermark "$watermark_type" --run_name copypaste-3-"$cp_attack_insertion_len"-neweva-"$watermark_type"-arxiv-"$model"  --input_dir /data/jiacheng/sok-llm-watermark/runs/"$token"/"$watermark_type"/arxiv/"$model"/copypaste-3-"$cp_attack_insertion_len" --wandb_project temporary --overwrite_args True --evaluation_metrics z-score
        done



        # # swap
        for attack_type in "${attack_types[@]}"; do
            # 对每个元素执行CUDA_VISIBLE_DEVICES=0,2 python命令
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True   --overwrite_output_file True --watermark "$watermark_type" --run_name "$attack_type"-neweva-"$watermark_type"-arxiv-"$model"  --input_dir /data/jiacheng/sok-llm-watermark/runs/"$token"/"$watermark_type"/arxiv/"$model"/"$attack_type" --wandb_project eva-aiwei23b-attack --overwrite_args True --evaluation_metrics z-score
        done


        # #dipper need 48gb memory
        for dipper_lex in "${dipper_lexs[@]}"; do
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True   --overwrite_output_file True --watermark "$watermark_type" --run_name dipper-neweva-"$watermark_type"-arxiv-"$model"-dipper_l"$dipper_lex"_o0   --input_dir /data/jiacheng/sok-llm-watermark/runs/"$token"/"$watermark_type"/arxiv/"$model"/dipper_l"$dipper_lex"_o0   --wandb_project eva-arxiv-attack --overwrite_args True --evaluation_metrics z-score
        done
        # CUDA_VISIBLE_DEVICES=0,2 python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True   --overwrite_output_file True --watermark "$watermark_type" --run_name dipper-neweva-"$watermark_type"-arxiv-"$model"-dipper_l60_o20   --input_dir /data/jiacheng/sok-llm-watermark/runs/"$token"/"$watermark_type"/arxiv/"$model"/dipper_l60_o20  --wandb_project eva-aiwei23b-attack --overwrite_args True

                # # #helm
        for helm_attack_method in "${helm_attack_methods[@]}"; do
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True   --overwrite_output_file True --watermark "$watermark_type" --run_name helm-"$helm_attack_method"-neweva-"$watermark_type"-arxiv-"$model"  --input_dir /data/jiacheng/sok-llm-watermark/runs/"$token"/"$watermark_type"/arxiv/"$model"/"$helm_attack_method"  --wandb_project eva-arxiv-attack --overwrite_args True --evaluation_metrics z-score
        done
    done
    done
    echo "finish $model"
done
echo "finish $watermark_type"


