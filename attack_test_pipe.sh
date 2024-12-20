#!/bin/bash
# 定义一个包含不同watermark类型的数组
# SUPPORTED_ATTACK_METHODS = [ "dipper"  "copy-paste"  "scramble" "helm" "oracle" "swap" "synonym"]
watermark_type="john23"
models=("opt" "llama")
# attack_types=( "dipper"  "copy-paste"  "scramble" "helm" "oracle" "swap" "synonym")
attack_types=( "swap")
synonym_probs=("0.1" "0.2" "0.4" "0.8" "1.0")
helm_attack_methods=("MisspellingAttack" "TypoAttack" "ContractionAttack" "LowercaseAttack" "ExpansionAttack")
dipper_lexs=("20" "40" "60")
cp_attack_insertion_lens=("10" "25")
# cp_attack_types=("single-single" "triple-single")
# 遍历数组中的每个元素
for model in "${models[@]}"; do
  

    for attack_type in "${attack_types[@]}"; do
        # 对每个元素执行python命令
        # python main.py --watermark "$attack_type"
        echo python watermark_reliability_release/attack_pipeline.py --run_name "$attack_type"-attack-"$watermark_type"-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/"$watermark_type"/c4/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/"$watermark_type"/c4/"$model"/"$attack_type" --attack_method "$attack_type"
    done


    #synonym
    for synonym_prob in "${synonym_probs[@]}"; do
        echo python watermark_reliability_release/attack_pipeline.py --run_name synonym-"$synonym_prob"-attack-"$watermark_type"-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/"$watermark_type"/c4/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/"$watermark_type"/c4/"$model"/synonym-"$synonym_prob" --attack_method synonym --synonym_p "$synonym_prob"
    done

    #copypaste
    for cp_attack_insertion_len in "${cp_attack_insertion_lens[@]}"; do
        echo python watermark_reliability_release/attack_pipeline.py --cp_attack_types single-single --run_name copypaste-1-"$cp_attack_insertion_len"-attack-"$watermark_type"-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/"$watermark_type"/c4/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/"$watermark_type"/c4/"$model"/copypaste-1-"$cp_attack_insertion_len" --attack_method copy-paste --cp_attack_insertion_len "$cp_attack_insertion_len"
        echo python watermark_reliability_release/attack_pipeline.py --cp_attack_types triple-single --run_name copypaste-3-"$cp_attack_insertion_len"-attack-"$watermark_type"-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/"$watermark_type"/c4/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/"$watermark_type"/c4/"$model"/copypaste-3-"$cp_attack_insertion_len" --attack_method copy-paste --cp_attack_insertion_len "$cp_attack_insertion_len"
    done

    #helm
    for helm_attack_method in "${helm_attack_methods[@]}"; do
        echo python watermark_reliability_release/attack_pipeline.py --run_name helm-"$helm_attack_method"-attack-"$watermark_type"-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/"$watermark_type"/c4/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/"$watermark_type"/c4/"$model"/"$attack_type" --attack_method helm --helm_attack_method "$helm_attack_method"
    done

    #dipper need 48gb memory
    for dipper_lex in "${dipper_lexs[@]}"; do
        echo python watermark_reliability_release/attack_pipeline.py --run_name dipper-attack-"$watermark_type"-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/"$watermark_type"/c4/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/"$watermark_type"/c4/"$model"/dipper_l"$dipper_lex"_o0 --attack_method dipper --lex "$dipper_lex"
    done
    echo python watermark_reliability_release/attack_pipeline.py --run_name dipper-attack-"$watermark_type"-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/"$watermark_type"/c4/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/"$watermark_type"/c4/"$model"/dipper_l60_o20 --attack_method dipper --lex 60 --order 20

done