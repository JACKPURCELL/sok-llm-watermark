
# Carved in Stone or Written in Sand? A Robustness Assessment of Language Model Watermarking
## Code is being cleaned up, please use it later
## Support Watermark
```shell
TGRL(john23)
UG(xuandong23b)
UPV(aiwei23)
SIR(aiwei23b)
RDF(rohith23)
UB(xiaoniu23)
GO(scott22)
CTWL(lean23)
MPAC(kiyoon23)
```
## Apply the permission of LLAMA2 in huggingface
```shell
https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
```
You can replace to this if you have no permission
```shell
https://huggingface.co/daryl149/llama-2-7b-chat-hf
```
## Sign up Wandb
```shell
#https://wandb.ai/alps-lab-sok

wandb login
#Input your wandb key
```
## Environment
```shell
#Python 3.11
conda create -n sokwm python=3.11
conda activate sokwm
pip install -r watermark_reliability_release/requirements.txt
```

## Argument Parsers
Set up the argument parsers for the minimum and maximum number of generations and tokens.
```python
parser.add_argument(
    "--min_generations",
    type=int,
    default=500,
    help="The minimum number of valid generations according to the output check strat to sample."
)

parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=100,
    help="The number of tokens to generate using the model, and the num tokens removed from real text sample"
)
```
## Watermark Injection
Use two GPUs for watermark injection. Commands for running the watermark injection pipeline.

- For watermark 'john23':
```shell
  CUDA_VISIBLE_DEVICES=2,0,1,3 python watermark_reliability_release/generation_pipeline.py --watermark john23 --dataset_name c4 --run_name gen-c4-john23 --model_name_or_path meta-llama/Llama-2-7b-chat-hf --min_generations 10
```
- Put it to background:
```shell
  CUDA_VISIBLE_DEVICES=2,0,1,3 nohup python watermark_reliability_release/generation_pipeline.py --watermark john23 --dataset_name c4 --run_name gen-c4-john23 --model_name_or_path meta-llama/Llama-2-7b-chat-hf --min_generations 10 &
```
- For watermark 'xuandong23b':
```shell
  CUDA_VISIBLE_DEVICES=1,0,2,3 python watermark_reliability_release/generation_pipeline.py --watermark xuandong23b --dataset_name c4 --run_name gen-c4-xuandong23b --model_name_or_path meta-llama/Llama-2-7b-chat-hf --min_generations 10
```
## Attack Phase (Optional)
Attack phase can be skipped. Requires 1 GPU with 46GB memory. Commands for running the attack pipeline.

- Run all the basic attack
```shell
nohup bash release_shell/attack_john23.sh > release_shell/attack_john23.log 2>&1 &
```

- For 'john23' watermark:
```shell
  CUDA_VISIBLE_DEVICES=2 python watermark_reliability_release/attack_pipeline.py --run_name dipper-attack-john23-c4 --input_dir ~/sok-llm-watermark/runs/john23/c4 --output_dir ~/sok-llm-watermark/runs/john23/c4/dipper --attack_method dipper
  CUDA_VISIBLE_DEVICES=2 python watermark_reliability_release/attack_pipeline.py --run_name cp-attack-john23-c4 --input_dir ~/sok-llm-watermark/runs/john23/c4 --output_dir ~/sok-llm-watermark/runs/john23/c4/copypaste --attack_method copy-paste
```
- For 'xuandong23b' watermark:
```shell
  CUDA_VISIBLE_DEVICES=0 python watermark_reliability_release/attack_pipeline.py --run_name dipper-attack-xuandong23b-c4 --input_dir ~/sok-llm-watermark/runs/xuandong23b/c4 --output_dir ~/sok-llm-watermark/runs/xuandong23b/c4/dipper --attack_method dipper
  CUDA_VISIBLE_DEVICES=1 python watermark_reliability_release/attack_pipeline.py --run_name cp-attack-xuandong23b-c4 --input_dir ~/sok-llm-watermark/runs/xuandong23b/c4 --output_dir ~/sok-llm-watermark/runs/xuandong23b/c4/copypaste --attack_method copy-paste
```
## Evaluation (Optional)
Requires 1 GPU with 8GB memory. Commands for running the evaluation pipeline.

- Evaluation all the basic attack
```shell
nohup bash release_shell/eva_john23.sh > release_shell/eva_john23.log 2>&1 &
```

- For 'john23' watermark:
```shell
  CUDA_VISIBLE_DEVICES=2 python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark john23 --run_name eva-john23-c4-dipper --input_dir ~/sok-llm-watermark/runs/john23/c4/dipper --evaluation_metrics all
  CUDA_VISIBLE_DEVICES=2 python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark john23 --run_name eva-john23-c4 --input_dir ~/sok-llm-watermark/runs/john23/c4 --evaluation_metrics all
```
- For 'xuandong23b' watermark:
```shell
  CUDA_VISIBLE_DEVICES=0 python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark xuandong23b --run_name eva-xuandong23b-c4-dipper --input_dir ~/sok-llm-watermark/runs/xuandong23b/c4/dipper --evaluation_metrics all
  CUDA_VISIBLE_DEVICES=1 python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark xuandong23b --run_name eva
```

## Generic Detector (Optional)

### Finetune generic detector
```shell
  CUDA_VISIBLE_DEVICES=0,1 python /home/jkl6486/sok-llm-watermark/generic_detector/fine_tune_roberta_gpt_data.py --train_path /home/jkl6486/sok-llm-watermark/runs/token_200/john23/c4/opt/dipper_40_o0/gen_table_w_metrics.jsonl  --output_path /home/jkl6486/sok-llm-watermark/runs/token_200/john23/c4/opt/dipper_40_o0/dipp_roberta_finetuned_chatgpt_new  --method_name john23

  python generic_detector/roberta_detect.py
```

### Adversarial attack
```shell
  CUDA_VISIBLE_DEVICES=0,1 python whitebox_attack.py --method  john23

  python generic_detector/roberta_detect_perturb.py
```
### Multi GPT attack
```shell
  nohup bash release_shell/multi-gpt-attack.sh > release_shell/multi-gpt-attack.log 2>&1 &
```