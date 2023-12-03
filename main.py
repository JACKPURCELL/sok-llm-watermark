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
import pickle

import torch
from tqdm import tqdm

import watermarks
import attacks
from utils.logger import MetricLogger
from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          LogitsProcessorList)
os.environ['TRANSFORMERS_CACHE'] = '/data/jc/model'

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
        default="facebook/opt-6.7b",
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
    parser.add_argument(
        "--skip_inject",
        type=str2bool,
        default=False,
        help="Skip the model loading to debug the interface.",
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

    #aiwei23
    parser.add_argument("--bit_number", type=int, default=16) ### This is log2(vocab_size), which depends on the model, for opt, it is 16
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--window_size", type=int, default=3)
    parser.add_argument("--llm_name", type=str, default="opt-6.7b")
    parser.add_argument("--gamma", type=float, default=0.6)
    parser.add_argument("--delta", type=float, default= 2.0)
    parser.add_argument("--model_dir", type=str, default="./model/")
    parser.add_argument("--beam_size", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--z_value", type=int, default=4)
    parser.add_argument("--sample_number", type=int, default=2000, help="Number of samples for training generator.")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples for training detector.")
    parser.add_argument("--dataset_name", type=str, default="c4", help="The dataset used for training detector.")
    parser.add_argument("--sampling_temp", type=float, default=0.7)
    parser.add_argument("--max_new_token", type=int, default=200)
    parser.add_argument("--aiwei_trained", type=bool, default=False)

    ######################################################################
    # Add your code here
    ######################################################################
    # If you have specific arguments for your watermark, add them here
    ######################################################################
    
    
    parser.add_argument("--attack", type=str)
    parser.add_argument("--log_dir", type=str)
    
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


from sklearn.metrics import roc_auc_score
from jiwer import wer
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import warnings
def compare(gt_text,  modify_text, args, device=None, tokenizer=None, **kwargs):
    # Initialize results dictionary


    # Word Error Rate TODO: make sure this metric work for us
    # Note: WER is calculated between the ground truth and the watermarked text.
    if isinstance(gt_text, str):
        gt_text = [gt_text]
    word_error_rate = wer(gt_text[0], modify_text)

    # BERT-S (BERT Score)
    # Note: BERT Score requires tokenized sentences. Adjust the code to fit your context.
    # Some weights of the model checkpoint at roberta-large were not used when initializing RobertaModel: ['lm_head.layer_norm.bias', 'lm_head.layer_norm.weight', 'lm_head.dense.bias', 'lm_head.bias', 'lm_head.dense.weight']
# - This IS expected if you are initializing RobertaModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
# - This IS NOT expected if you are initializing RobertaModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
# Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        P, R, F1 = bert_score([modify_text], [gt_text], lang='en', device=device)
    BERT_S = F1.mean().item() # Considering F1 here. Adjust as needed.

    # BLEU-4
    # Note: BLEU-4 requires tokenized sentences. Adjust as per your data.
    reference = []
    for i in gt_text:
        reference.append(tokenizer.tokenize(i))
    candidate = tokenizer.tokenize(modify_text)
    smoothing = SmoothingFunction().method1
    BLEU_4 = sentence_bleu(reference, candidate, smoothing_function=smoothing)

    return  word_error_rate, BERT_S, BLEU_4

# def evaluate(gt_text, modify_text, args, device=None, tokenizer=None, **kwargs):
    
    
def attack(args, dp=None, prompt=None, output_text=None):

    output2 = None
    match args.attack:
        case 'dipper':
            output = dp.paraphrase(output_text, lex_diversity=40, order_diversity=0, prefix=prompt, do_sample=True, top_p=0.75, top_k=None, max_length=512)
            # output_l60_o60_greedy
            # output = dp.paraphrase(output_text, lex_diversity=80, order_diversity=60, prefix=prompt, do_sample=False, max_length=512)
            # output_l60_sample
            # output2 = dp.paraphrase(output_text, lex_diversity=80, order_diversity=0, prefix=prompt, do_sample=True, top_p=0.75, top_k=None, max_length=512)
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
    logger = MetricLogger()
    if args.skip_inject:
        tokenizer,device = load_tokenizer(args)
    else:
        model, tokenizer, device = load_model(args)

    if args.log_dir is None:
        log_dir = 'runs/'+str(args.watermark)
    else:
        log_dir = 'runs/'+args.log_dir
    
    try:
        os.mkdir(log_dir)
    except:
        print( "Directory %s already exists" % log_dir)


    with open(os.path.join(log_dir,'args.txt'), mode='w') as f:
        json.dump(args.__dict__, f, indent=2)
        

        
    # Generate and detect, report to stdout
    if not args.skip_inject:
        
        with open(os.path.join(log_dir, "output.json"), "w+") as f:
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
            case 'rohith23':
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

            case 'aiwei23':
                if args.aiwei_trained:
                    watermark_detector = pickle.load(open('./data/trained/trained_detector.pkl', 'rb'))
                    watermark_processor = pickle.load(open('./data/trained/trained_processor.pkl', 'rb'))
                else:
                    watermarks.prepare_generator(bit_number=args.bit_number,
                                      layers=args.layers,
                                      sample_number=args.sample_number,
                                      window_size=args.window_size)
                    watermark_detector = watermarks.aiwei23_WatermarkDetector(bit_number=args.bit_number,
                                                         window_size=args.window_size,
                                                         layers=args.layers,
                                                         gamma=args.gamma,
                                                         delta=args.delta,
                                                         model_dir=args.model_dir,
                                                         beam_size=args.beam_size,
                                                         llm_name=args.llm_name,
                                                         data_dir=args.data_dir,
                                                         z_value=args.z_value)

                    watermark_detector.generate_and_save_train_data(num_samples=args.num_samples)
                    watermark_processor = watermark_detector.generate_and_save_test_data(dataset_name=args.dataset_name,
                                                                               sampling_temp=args.sampling_temp,
                                                                               max_new_tokens=args.max_new_token)
                    watermark_detector.train_model()
                    pickle.dump(watermark_detector, open('./data/trained/trained_detector.pkl', 'wb'))
                    pickle.dump(watermark_processor, open('./data/trained/trained_processor.pkl', 'wb'))

    
            ######################################################################
            # Add your code here
            ######################################################################
            # If you have new watermark, add them here
            ######################################################################   
        
            case _:
                raise ValueError(f"Unknown watermark type: {args.watermark}")
 
        for idx, cur_data in enumerate(tqdm(data)):
            prompt = cur_data['question']
            human_answers = []
            for human_answer in cur_data['human_answers']:
                human_answers.append(human_answer)


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

          
            if with_watermark_detection_result['prediction'] == True:
                logger.update(TPR_count=1)
            else:
                logger.update(TPR_count=0)
            if without_watermark_detection_result['prediction'] == True:
                logger.update(FPR_count=1)
            else:
                logger.update(FPR_count=0)
            output = {
                "prompt": prompt,
                "human_answers": human_answers,
                "decoded_output_without_watermark": decoded_output_without_watermark,
                "decoded_output_with_watermark": decoded_output_with_watermark,
                "without_watermark_detection_result": without_watermark_detection_result,
                "with_watermark_detection_result": with_watermark_detection_result,
            }
            with open(os.path.join(log_dir, "output.json"), "a+") as f:
                json.dump(output, f)
                f.write('\n')
            result_hm_wo = compare(human_answers,decoded_output_without_watermark,args,device,tokenizer)
            logger.update(hm_wo_word_error_rate=result_hm_wo[0], hm_wo_BERT_S=result_hm_wo[1], hm_wo_BLEU_4=result_hm_wo[2])
            result_hm_wm = compare(human_answers,decoded_output_with_watermark,args,device,tokenizer)
            logger.update(hm_wm_word_error_rate=result_hm_wm[0], hm_wm_BERT_S=result_hm_wm[1], hm_wm_BLEU_4=result_hm_wm[2])
            result_wo_wm = compare(decoded_output_without_watermark,decoded_output_with_watermark,args,device,tokenizer)
            logger.update(wo_wm_word_error_rate=result_wo_wm[0], wo_wm_BERT_S=result_wo_wm[1], wo_wm_BLEU_4=result_wo_wm[2])
        hm_wo_word_error_rate, hm_wo_BERT_S, hm_wo_BLEU_4 = logger.meters['hm_wo_word_error_rate'].global_avg, logger.meters['hm_wo_BERT_S'].global_avg, logger.meters['hm_wo_BLEU_4'].global_avg
        hm_wm_word_error_rate, hm_wm_BERT_S, hm_wm_BLEU_4 = logger.meters['hm_wm_word_error_rate'].global_avg, logger.meters['hm_wm_BERT_S'].global_avg, logger.meters['hm_wm_BLEU_4'].global_avg
        wo_wm_word_error_rate, wo_wm_BERT_S, wo_wm_BLEU_4 = logger.meters['wo_wm_word_error_rate'].global_avg, logger.meters['wo_wm_BERT_S'].global_avg, logger.meters['wo_wm_BLEU_4'].global_avg
        TPR_count,FPR_count = logger.meters['TPR_count'].global_avg, logger.meters['FPR_count'].global_avg
        results = {
        'hm_wo_word_error_rate': hm_wo_word_error_rate,
        'hm_wo_BERT_S': hm_wo_BERT_S,
        'hm_wo_BLEU_4': hm_wo_BLEU_4,
        'hm_wm_word_error_rate': hm_wm_word_error_rate,
        'hm_wm_BERT_S': hm_wm_BERT_S,
        'hm_wm_BLEU_4': hm_wm_BLEU_4,
        'wo_wm_word_error_rate': wo_wm_word_error_rate,
        'wo_wm_BERT_S': wo_wm_BERT_S,
        'wo_wm_BLEU_4': wo_wm_BLEU_4,
        'TPR_count': TPR_count,
        'FPR_count': FPR_count
        }

        with open(os.path.join(log_dir,"log.csv"), 'w') as f:
            # 第一行是所有的键，用逗号分隔
            f.write(','.join(results.keys()) + '\n')
            # 第二行是所有的值，用逗号分隔
            f.write(','.join(map(str, results.values())) + '\n')
            
        
    if args.attack is not None:
        with open(os.path.join(log_dir,args.attack+".json"), "w+") as f:
            pass
        data = read_json_file(os.path.join(log_dir, "output.json"))
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
            case 'rohith23':
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

            case 'aiwei23':
                if args.aiwei_trained:
                    watermark_detector = pickle.load(open('./data/trained/trained_detector.pkl', 'rb'))
                    watermark_processor = pickle.load(open('./data/trained/trained_processor.pkl', 'rb'))
                else:
                    watermarks.prepare_generator(bit_number=args.bit_number,
                                                 layers=args.layers,
                                                 sample_number=args.sample_number,
                                                 window_size=args.window_size)
                    watermark_detector = watermarks.aiwei23_WatermarkDetector(bit_number=args.bit_number,
                                                                              window_size=args.window_size,
                                                                              layers=args.layers,
                                                                              gamma=args.gamma,
                                                                              delta=args.delta,
                                                                              model_dir=args.model_dir,
                                                                              beam_size=args.beam_size,
                                                                              llm_name=args.llm_name,
                                                                              data_dir=args.data_dir,
                                                                              z_value=args.z_value)

                    watermark_detector.generate_and_save_train_data(num_samples=args.num_samples)
                    watermark_processor = watermark_detector.generate_and_save_test_data(dataset_name=args.dataset_name,
                                                                                         sampling_temp=args.sampling_temp,
                                                                                         max_new_tokens=args.max_new_token)
                    watermark_detector.train_model()
                    pickle.dump(watermark_detector, open('./data/trained/trained_detector.pkl', 'wb'))
                    pickle.dump(watermark_processor, open('./data/trained/trained_processor.pkl', 'wb'))

           
        
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
                
            
        for idx, cur_data in enumerate(tqdm(data)):
            prompt = cur_data["prompt"]
            human_answers = []
            for human_answer in cur_data['human_answers']:
                human_answers.append(human_answer)
            decoded_output_with_watermark = cur_data["decoded_output_with_watermark"]
            
            attack_output, output2 = attack(args, dp, prompt, decoded_output_with_watermark)
            
            after_attack_detection_result,_ = detect(watermark_detector,
                                        attack_output, 
                                        prompt,
                                        args, 
                                        device=device, 
                                        tokenizer=tokenizer)
            if after_attack_detection_result['prediction'] == True:
                logger.update(TPR_count=1)
            else:
                logger.update(TPR_count=0)
            cur_data["attack_output"] = attack_output
            cur_data["attack_output_result"] = after_attack_detection_result
            result_hm_at = compare(human_answers,attack_output,args,device,tokenizer)
            result_wm_at = compare(decoded_output_with_watermark,attack_output,args,device,tokenizer)
            logger.update(hm_at_word_error_rate=result_hm_at[0], hm_at_BERT_S=result_hm_at[1], hm_at_BLEU_4=result_hm_at[2])
            logger.update(wm_at_word_error_rate=result_wm_at[0], wm_at_BERT_S=result_wm_at[1], wm_at_BLEU_4=result_wm_at[2])
            
            with open(os.path.join(log_dir,args.attack+".json"), "a+") as f:
                json.dump(cur_data, f)
                f.write('\n')
        hm_at_word_error_rate, hm_at_BERT_S, hm_at_BLEU_4 = logger.meters['hm_at_word_error_rate'].global_avg, logger.meters['hm_at_BERT_S'].global_avg, logger.meters['hm_at_BLEU_4'].global_avg
        wm_at_word_error_rate, wm_at_BERT_S, wm_at_BLEU_4 = logger.meters['wm_at_word_error_rate'].global_avg, logger.meters['wm_at_BERT_S'].global_avg, logger.meters['wm_at_BLEU_4'].global_avg
        results = {
            'hm_at_word_error_rate': hm_at_word_error_rate,
            'hm_at_BERT_S': hm_at_BERT_S,
            'hm_at_BLEU_4': hm_at_BLEU_4,
            'wm_at_word_error_rate': wm_at_word_error_rate,
            'wm_at_BERT_S': wm_at_BERT_S,
            'wm_at_BLEU_4': wm_at_BLEU_4,
        }

        with open(os.path.join(log_dir,"log.csv"), 'a+') as f:
            # 第一行是所有的键，用逗号分隔
            f.write(','.join(results.keys()) + '\n')
            # 第二行是所有的值，用逗号分隔
            f.write(','.join(map(str, results.values())) + '\n')
            
            
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