# coding=utf-8
# Copyright 2023 Authors of "A Watermark for Large Language Models" 
# available at https://arxiv.org/abs/2301.10226
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
from argparse import Namespace
from pprint import pprint
from functools import partial
import json
import numpy # for gradio hot reload
import gradio as gr

import torch
from tqdm import tqdm

import watermarks
import attacks

from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          LogitsProcessorList)

# from watermark_processor import WatermarkLogitsProcessor, WatermarkDetector

def str2bool(v):
    """Util function for user friendly boolean flag args"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    """Command line argument specification"""

    parser = argparse.ArgumentParser(description="A minimum working example of applying the watermark to any LLM that supports the huggingface 🤗 `generate` API")

    parser.add_argument(
        "--watermark",
        type=str,
        default='',
        help="Select the watermark type",
    )
    parser.add_argument(
        "--run_gradio",
        type=str2bool,
        default=True,
        help="Whether to launch as a gradio demo. Set to False if not installed and want to just run the stdout version.",
    )
    parser.add_argument(
        "--demo_public",
        type=str2bool,
        default=False,
        help="Whether to expose the gradio demo to the internet.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/opt-1.3b",
        help="Main model, path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--prompt_max_length",
        type=int,
        default=None,
        help="Truncation length for prompt, overrides model config's max length field.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Maximmum number of new tokens to generate.",
    )
    parser.add_argument(
        "--generation_seed",
        type=int,
        default=123,
        help="Seed for setting the torch global rng prior to generation.",
    )
    parser.add_argument(
        "--use_sampling",
        type=str2bool,
        default=True,
        help="Whether to generate using multinomial sampling.",
    )
    parser.add_argument(
        "--sampling_temp",
        type=float,
        default=0.7,
        help="Sampling temperature to use when generating using multinomial sampling.",
    )
    parser.add_argument(
        "--n_beams",
        type=int,
        default=1,
        help="Number of beams to use for beam search. 1 is normal greedy decoding",
    )
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default=True,
        help="Whether to run inference and watermark hashing/seeding/permutation on gpu.",
    )
    parser.add_argument(
        "--seeding_scheme",
        type=str,
        default="simple_1",
        help="Seeding scheme to use to generate the greenlists at each generation and verification step.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.25,
        help="The fraction of the vocabulary to partition into the greenlist at each generation and verification step.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=2.0,
        help="The amount/bias to add to each of the greenlist token logits before each token sampling step.",
    )
    parser.add_argument(
        "--normalizers",
        type=str,
        default="",
        help="Single or comma separated list of the preprocessors/normalizer names to use when performing watermark detection.",
    )
    parser.add_argument(
        "--ignore_repeated_bigrams",
        type=str2bool,
        default=False,
        help="Whether to use the detection method that only counts each unqiue bigram once as either a green or red hit.",
    )
    parser.add_argument(
        "--detection_z_threshold",
        type=float,
        default=4.0,
        help="The test statistic threshold for the detection hypothesis test.",
    )
    parser.add_argument(
        "--select_green_tokens",
        type=str2bool,
        default=True,
        help="How to treat the permuation when selecting the greenlist tokens at each step. Legacy is (False) to pick the complement/reds first.",
    )
    parser.add_argument(
        "--skip_model_load",
        type=str2bool,
        default=False,
        help="Skip the model loading to debug the interface.",
    )
    parser.add_argument(
        "--seed_separately",
        type=str2bool,
        default=True,
        help="Whether to call the torch seed function before both the unwatermarked and watermarked generate calls.",
    )
    parser.add_argument(
        "--load_fp16",
        type=str2bool,
        default=False,
        help="Whether to run model in float16 precsion.",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default='/home/jkl6486/sok-llm-watermark/dataset/sample.jsonl'
    )
    
    #xuandong23b
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--strength", type=float, default=2.0)
    parser.add_argument("--wm_key", type=int, default=0)

    #lean23
    parser.add_argument("--lean_delta", type=float, default=1.5)
    parser.add_argument("--lm_prefix_len", type=int, default=10)
    parser.add_argument("--lm_top_k", type=int, default=-1),
    parser.add_argument("--message_code_len", type=int, default=20)
    parser.add_argument("--random_permutation_num", type=int, default=100)
    parser.add_argument("--encode_ratio", type=float, default=10.0)
    parser.add_argument("--max_confidence_lbd", type=float, default=0.5)
    parser.add_argument("--message_model_strategy", type=str, default="vanilla")
    parser.add_argument("--message", type=list, default=[9,10,100])
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--repeat_penalty", type=float, default=1.5)
    parser.add_argument("--generated_length", type=int, default=200)
    parser.add_argument("--prompt_length", type=int, default=300)

    
    
    ######################################################################
    # Add your code here
    ######################################################################
    # If you have specific arguments for your watermark, add them here
    ######################################################################
    
    
    parser.add_argument("--attack", type=str)
    
    args = parser.parse_args()
    return args

def load_model(args):
    """Load and return the model and tokenizer"""

    args.is_seq2seq_model = any([(model_type in args.model_name_or_path) for model_type in ["t5","T0"]])
    args.is_decoder_only_model = any([(model_type in args.model_name_or_path) for model_type in ["gpt","opt","bloom"]])
    if args.is_seq2seq_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    elif args.is_decoder_only_model:
        if args.load_fp16:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,torch_dtype=torch.float16, device_map='auto')
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(f"Unknown model type: {args.model_name_or_path}")

    if args.use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.load_fp16: 
            pass
        else: 
            model = model.to(device)
    else:
        device = "cpu"
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    return model, tokenizer, device

def load_tokenizer(args):
    """Load and return the model and tokenizer"""

    args.is_seq2seq_model = any([(model_type in args.model_name_or_path) for model_type in ["t5","T0"]])
    args.is_decoder_only_model = any([(model_type in args.model_name_or_path) for model_type in ["gpt","opt","bloom"]])


    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if args.use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"
        
    return tokenizer, device

def generate(watermark_processor, prompt, args, model=None, device=None, tokenizer=None):
    """Instatiate the WatermarkLogitsProcessor according to the watermark parameters
       and generate watermarked text by passing it to the generate method of the model
       as a logits processor. """
   
            
            
    # watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
    #                                                 gamma=args.gamma,
    #                                                 delta=args.delta,
    #                                                 seeding_scheme=args.seeding_scheme,
    #                                                 select_green_tokens=args.select_green_tokens)

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens)

    if args.use_sampling:
        gen_kwargs.update(dict(
            do_sample=True, 
            top_k=0,
            temperature=args.sampling_temp
        ))
    else:
        gen_kwargs.update(dict(
            num_beams=args.n_beams
        ))

    generate_without_watermark = partial(
        model.generate,
        **gen_kwargs
    )
    generate_with_watermark = partial(
        model.generate,
        logits_processor=LogitsProcessorList([watermark_processor]), 
        **gen_kwargs
    )
    if args.prompt_max_length:
        pass
    elif hasattr(model.config,"max_position_embedding"):
        args.prompt_max_length = model.config.max_position_embeddings-args.max_new_tokens
    else:
        args.prompt_max_length = 2048-args.max_new_tokens

    tokd_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=args.prompt_max_length).to(device)
    truncation_warning = True if tokd_input["input_ids"].shape[-1] == args.prompt_max_length else False
    redecoded_input = tokenizer.batch_decode(tokd_input["input_ids"], skip_special_tokens=True)[0]

    torch.manual_seed(args.generation_seed)
    output_without_watermark = generate_without_watermark(**tokd_input)

    # optional to seed before second generation, but will not be the same again generally, unless delta==0.0, no-op watermark
    if args.seed_separately: 
        torch.manual_seed(args.generation_seed)
    output_with_watermark = generate_with_watermark(**tokd_input)

    if args.is_decoder_only_model:
        # need to isolate the newly generated tokens
        output_without_watermark = output_without_watermark[:,tokd_input["input_ids"].shape[-1]:]
        output_with_watermark = output_with_watermark[:,tokd_input["input_ids"].shape[-1]:]

    decoded_output_without_watermark = tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0]
    decoded_output_with_watermark = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0]

    return (redecoded_input,
            int(truncation_warning),
            decoded_output_without_watermark, 
            decoded_output_with_watermark,
            args) ### return watermark_processor
            # decoded_output_with_watermark)

def format_names(s):
    """Format names for the gradio demo interface"""
    s=s.replace("num_tokens_scored","Tokens Counted (T)")
    s=s.replace("num_green_tokens","# Tokens in Greenlist")
    s=s.replace("green_fraction","Fraction of T in Greenlist")
    s=s.replace("z_score","z-score")
    s=s.replace("p_value","p value")
    s=s.replace("prediction","Prediction")
    s=s.replace("confidence","Confidence")
    return s

def list_format_scores(score_dict, detection_threshold):
    """Format the detection metrics into a gradio dataframe input format"""
    lst_2d = []
    # lst_2d.append(["z-score threshold", f"{detection_threshold}"])
    for k,v in score_dict.items():
        if k=='green_fraction': 
            lst_2d.append([format_names(k), f"{v:.1%}"])
        elif k=='confidence': 
            lst_2d.append([format_names(k), f"{v:.3%}"])
        elif isinstance(v, float): 
            lst_2d.append([format_names(k), f"{v:.3g}"])
        elif isinstance(v, bool):
            lst_2d.append([format_names(k), ("Watermarked" if v else "Human/Unwatermarked")])
        else: 
            lst_2d.append([format_names(k), f"{v}"])
    if "confidence" in score_dict:
        lst_2d.insert(-2,["z-score Threshold", f"{detection_threshold}"])
    else:
        lst_2d.insert(-1,["z-score Threshold", f"{detection_threshold}"])
    return lst_2d


def detect(watermark_detector, input_text, prompt, args, device=None, tokenizer=None):
    """Instantiate the WatermarkDetection object and call detect on
        the input text returning the scores and outcome of the test"""
 

    # watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
    #                                     gamma=args.gamma,
    #                                     seeding_scheme=args.seeding_scheme,
    #                                     device=device,
    #                                     tokenizer=tokenizer,
    #                                     z_threshold=args.detection_z_threshold,
    #                                     normalizers=args.normalizers,
    #                                     ignore_repeated_bigrams=args.ignore_repeated_bigrams,
    #                                     select_green_tokens=args.select_green_tokens)
    if len(input_text)-1 > watermark_detector.min_prefix_len:
        output = watermark_detector.detect(text=input_text,prompt=prompt)
        # output = str_format_scores(score_dict, watermark_detector.z_threshold)
        # output = list_format_scores(score_dict, watermark_detector.z_threshold)
    else:
        # output = (f"Error: string not long enough to compute watermark presence.")
        output = [["Error","string too short to compute metrics"]]
        output += [["",""] for _ in range(6)]
    return output, args


def evaluate(watermark_detector, input_text, args, device=None, tokenizer=None):
    pass

def attack(args, dp=None, prompt=None, output_text=None):

    output2 = None
    match args.attack:
        case 'dipper':
            # output_l60_o60_greedy
            output = dp.paraphrase(output_text, lex_diversity=80, order_diversity=60, prefix=prompt, do_sample=False, max_length=512)
            # output_l60_sample
            output2 = dp.paraphrase(output_text, lex_diversity=80, order_diversity=0, prefix=prompt, do_sample=True, top_p=0.75, top_k=None, max_length=512)
        ######################################################################
        # Add your attack code here
        ######################################################################
        # If you have new attack, add them here
        ######################################################################   
        case _:
            raise ValueError(f"Unknown attack type: {args.attack}")
                             
    return (output,
            output2) 
                
def read_json_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f.read().strip().split("\n")]
    
def main(args): 


    """Run a command line version of the generation and detection operations
        and optionally launch and serve the gradio demo"""
    # Initial arg processing and log
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])
    print(args)

    if args.attack is None:
        model, tokenizer, device = load_model(args)
    else:
        tokenizer,device = load_tokenizer(args)


    # Generate and detect, report to stdout
    if args.attack is None:
        
        with open("output.json", "w+") as f:
            pass
        data = read_json_file(args.prompt_file)

        outputs = []

        match args.watermark:
            case 'john23':
                watermark_processor = watermarks.john23_WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                        gamma=args.gamma,
                                                        delta=args.delta,
                                                        seeding_scheme=args.seeding_scheme,
                                                        select_green_tokens=args.select_green_tokens)
                watermark_detector = watermarks.john23_WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=args.gamma,
                                        seeding_scheme=args.seeding_scheme,
                                        device=device,
                                        tokenizer=tokenizer,
                                        z_threshold=args.detection_z_threshold,
                                        normalizers=args.normalizers,
                                        ignore_repeated_bigrams=args.ignore_repeated_bigrams,
                                        select_green_tokens=args.select_green_tokens)
            case 'xuandong23b':
                watermark_processor = watermarks.xuandong23b_WatermarkLogitsProcessor(fraction=args.fraction,
                                                        strength=args.strength,
                                                        vocab_size=model.config.vocab_size,
                                                        watermark_key=args.wm_key)
                watermark_detector = watermarks.xuandong23b_WatermarkDetector(fraction=args.fraction,
                                                        tokenizer=tokenizer,
                                                        strength=args.strength,
                                                        vocab_size=model.config.vocab_size,
                                                        watermark_key=args.wm_key)
            case 'rohit23':
                watermark_processor = watermarks.rohith23_WatermarkLogitsProcessor(vocab_size=model.config.vocab_size)
                watermark_detector = watermarks.rohith23_WatermarkDetector(vocab_size=model.config.vocab_size,tokenizer=tokenizer)
                
            case 'lean23':
                watermark_processor = watermarks.lean23_BalanceMarkingWatermarkLogitsProcessor(tokenizer=tokenizer,
                                                                         lm_tokenizer=tokenizer,
                                                                         lm_model=model,
                                                                         delta=args.lean_delta,
                                                                         lm_prefix_len=args.lm_prefix_len,
                                                                         lm_top_k=args.lm_top_k,
                                                                         message_code_len=args.message_code_len,
                                                                         random_permutation_num=args.random_permutation_num,
                                                                         encode_ratio=args.encode_ratio,
                                                                         max_confidence_lbd=args.max_confidence_lbd,
                                                                         message_model_strategy=args.message_model_strategy,
                                                                         message=args.message,
                                                                         top_k=args.top_k,
                                                                         repeat_penalty=args.repeat_penalty
                                                                         )
                watermark_detector = watermarks.lean23_WatermarkDetector(watermark_processor=watermark_processor,
                                                                     generated_length=args.generated_length,
                                                                     message_code_len=args.message_code_len,
                                                                     encode_ratio=args.encode_ratio,
                                                                     tokenizer=tokenizer,
                                                                     prompt_length=args.prompt_length,
                                                                     message=args.message
                                                                     )
    
            ######################################################################
            # Add your code here
            ######################################################################
            # If you have new watermark, add them here
            ######################################################################   
        
            case _:
                raise ValueError(f"Unknown watermark type: {args.watermark}")
        
        for idx, cur_data in tqdm(enumerate(data)):
            prompt = cur_data['question']


            # batch = tokenizer(prefix, truncation=True, return_tensors="pt")
            # num_tokens = len(batch['input_ids'][0])
     
            # term_width = 80
            # print("#"*term_width)
            # print("Prompt: ")
            # print(prompt)


            _, _, decoded_output_without_watermark, decoded_output_with_watermark, _ = generate(watermark_processor,
                                                                                                prompt, 
                                                                                                args, 
                                                                                                model=model, 
                                                                                                device=device, 
                                                                                                tokenizer=tokenizer)
            without_watermark_detection_result, _ = detect(watermark_detector,
                                                        decoded_output_without_watermark, 
                                                        prompt,
                                                        args, 
                                                        device=device, 
                                                        tokenizer=tokenizer)
            
            with_watermark_detection_result, _ = detect(watermark_detector,
                                                    decoded_output_with_watermark, 
                                                    prompt,
                                                    args, 
                                                    device=device, 
                                                    tokenizer=tokenizer)

            # print("#"*term_width)
            # print("Output without watermark:")
            # print(decoded_output_without_watermark)
            # print("-"*term_width)
            # print(f"Detection result @ {args.detection_z_threshold}:")
            # pprint(without_watermark_detection_result)
            # print("-"*term_width)

            # print("#"*term_width)
            # print("Output with watermark:")
            # print(decoded_output_with_watermark)
            # print("-"*term_width)
            # print(f"Detection result @ {args.detection_z_threshold}:")
            # pprint(with_watermark_detection_result)
            # print("-"*term_width)
            
            output = {
                "prompt": prompt,
                "decoded_output_without_watermark": decoded_output_without_watermark,
                "decoded_output_with_watermark": decoded_output_with_watermark,
                "without_watermark_detection_result": without_watermark_detection_result,
                "with_watermark_detection_result": with_watermark_detection_result,
            }
            with open("output.json", "a+") as f:
                json.dump(output, f)
                f.write('\n')

    if args.attack is not None:
        with open("output_attack.json", "w+") as f:
            pass
        data = read_json_file("output.json")
        match args.watermark:
            case 'john23':
                watermark_detector = watermarks.john23_WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=args.gamma,
                                        seeding_scheme=args.seeding_scheme,
                                        device=device,
                                        tokenizer=tokenizer,
                                        z_threshold=args.detection_z_threshold,
                                        normalizers=args.normalizers,
                                        ignore_repeated_bigrams=args.ignore_repeated_bigrams,
                                        select_green_tokens=args.select_green_tokens)
            case 'xuandong23b':
                watermark_detector = watermarks.xuandong23b_WatermarkDetector(fraction=args.fraction,
                                                        tokenizer=tokenizer,
                                                        strength=args.strength,
                                                        vocab_size=tokenizer.vocab_size,
                                                        watermark_key=args.wm_key)
            case 'rohit23':
                watermark_detector = watermarks.rohith23_WatermarkDetector(vocab_size=tokenizer.vocab_size,tokenizer=tokenizer)
        
            case 'lean23':
                model, tokenizer, device = load_model(args)
                watermark_processor = watermarks.lean23_BalanceMarkingWatermarkLogitsProcessor(tokenizer=tokenizer,
                                                                         lm_tokenizer=tokenizer,
                                                                         lm_model=model,
                                                                         delta=args.lean_delta,
                                                                         lm_prefix_len=args.lm_prefix_len,
                                                                         lm_top_k=args.lm_top_k,
                                                                         message_code_len=args.message_code_len,
                                                                         random_permutation_num=args.random_permutation_num,
                                                                         encode_ratio=args.encode_ratio,
                                                                         max_confidence_lbd=args.max_confidence_lbd,
                                                                         message_model_strategy=args.message_model_strategy,
                                                                         message=args.message,
                                                                         top_k=args.top_k,
                                                                         repeat_penalty=args.repeat_penalty
                                                                         )
                watermark_detector = watermarks.lean23_WatermarkDetector(watermark_processor=watermark_processor,
                                                                     generated_length=args.generated_length,
                                                                     message_code_len=args.message_code_len,
                                                                     encode_ratio=args.encode_ratio,
                                                                     tokenizer=tokenizer,
                                                                     prompt_length=args.prompt_length,
                                                                     message=args.message
                                                                     )

           
        
            case _:
                raise ValueError(f"Unknown watermark type: {args.watermark}")
            
        match args.attack:
            case 'dipper':
                dp = attacks.DipperParaphraser(model="kalpeshk2011/dipper-paraphraser-xxl")
        ######################################################################
        # Add your attack code here
        ######################################################################
        # If you have new attack, add them here
        ######################################################################   
            case _:
                raise ValueError(f"Unknown attack type: {args.attack}")
                
            
        for idx, cur_data in tqdm(enumerate(data)):
            prompt = cur_data["prompt"]
            decoded_output_with_watermark = cur_data["decoded_output_with_watermark"]
            
            attack_output, output2 = attack(args, dp, prompt, decoded_output_with_watermark)
            
            after_attack_detection_result,_ = detect(watermark_detector,
                                        attack_output, 
                                        prompt,
                                        args, 
                                        device=device, 
                                        tokenizer=tokenizer)
            
            cur_data["attack_output"] = attack_output
            cur_data["attack_output_result"] = after_attack_detection_result
            with open("output_attack.json", "a+") as f:
                json.dump(cur_data, f)
                f.write('\n')
                

    # Launch the app to generate and detect interactively (implements the hf space demo)
    # if args.run_gradio:
    #     run_gradio(args, model=model, tokenizer=tokenizer, device=device)

    return

if __name__ == "__main__":

    args = parse_args()
    # args.use_gpu = False   ### Tested on a laptop without gpu
    # args.prompt_file = "./dataset/sample.jsonl"
    # args.watermark = "john23"
    # args.attack = "dipper"
    # args.max_new_tokens = 100
    print(args)

    main(args)