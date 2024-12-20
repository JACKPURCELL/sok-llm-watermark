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

import openai
import random
import torch
from tqdm import tqdm

from utils.dipper_attack_pipeline import generate_dipper_paraphrases
from utils.generate_translation import generate_translation

from utils.evaluation import OUTPUT_TEXT_COLUMN_NAMES
from utils.copy_paste_attack import single_insertion, triple_insertion_single_len, k_insertion_t_len
import utils.helm_attack as helm
from utils.swap_attack import SwapAttack

# from utils.oracle_attack.attack import Attacker, Oracle, Trainer
import utils.oracle_attack.attack as oracle_att
from utils.synonym import SynonymAttack
# SUPPORTED_ATTACK_METHODS = ["gpt", "dipper", "copy-paste", "scramble"]
SUPPORTED_ATTACK_METHODS = ["gpt",  "dipper", "copy-paste", "scramble","helm","oracle","swap","synonym","translation"]


def scramble_attack(example, tokenizer=None, args=None):
    # check if the example is long enough to attack
    for column in ["w_wm_output", "no_wm_output"]:
        if not check_output_column_lengths(example, min_len=args.cp_attack_min_len):
            # # if not, copy the orig w_wm_output to w_wm_output_attacked
            # NOTE changing this to return "" so that those fail/we can filter out these examples
            example[f"{column}_attacked"] = ""
            example[f"{column}_attacked_length"] = 0
        else:
            sentences = example[column].split(".")
            random.shuffle(sentences)
            example[f"{column}_attacked"] = ".".join(sentences)
            example[f"{column}_attacked_length"] = len(
                tokenizer(example[f"{column}_attacked"])["input_ids"]
            )
    return example


def gpt_attack(example, attack_prompt=None, args=None):
    assert attack_prompt, "Prompt must be provided for GPT attack"

    gen_row = example

    if args.no_wm_attack:
        original_text = gen_row["no_wm_output"]
    else:
        original_text = gen_row["w_wm_output"]

    attacker_query = attack_prompt + original_text
    query_msg = {"role": "user", "content": attacker_query}


    from openai import OpenAI
    client = OpenAI()

    # response = client.chat.completions.create(
    # model="gpt-3.5-turbo-0125",
    # response_format={ "type": "json_object" },
    # messages=[
    #     {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
    #     {"role": "user", "content": "Who won the world series in 2020?"}
    # ]
    # )
    # print(response.choices[0].message.content)

    # from tenacity import retry, stop_after_attempt, wait_random_exponential

    # # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
    # @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    # def completion_with_backoff(model, messages, temperature, max_tokens):
    #     return openai.ChatCompletion.create(
    #         model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
    #     )

    from tenacity import (
        retry,
        stop_after_attempt,
        wait_random_exponential,
    )  # for exponential backoff

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def completion_with_backoff(**kwargs):
        return client.chat.completions.create(**kwargs)


    outputs = completion_with_backoff(model="gpt-3.5-turbo", messages=[query_msg])

    # outputs = completion_with_backoff(
    #     model=args.attack_model_name,
    #     messages=[query_msg],
    #     temperature=args.attack_temperature,
    #     max_tokens=args.attack_max_tokens,
    # )

    attacked_text = outputs.choices[0].message.content
    assert (
        len(outputs.choices) == 1
    ), "OpenAI API returned more than one response, unexpected for length inference of the output"
    example["w_wm_output_attacked_length"] = outputs.usage.completion_tokens
    example["w_wm_output_attacked"] = attacked_text
    if args.verbose:
        print(f"\nOriginal text (T={example['w_wm_output_length']}):\n{original_text}")
        print(f"\nAttacked text (T={example['w_wm_output_attacked_length']}):\n{attacked_text}")

    return example


def dipper_attack(dataset, lex=None, order=None, args=None):
    dataset = generate_dipper_paraphrases(dataset, lex=lex, order=order, args=args)
    return dataset

def translation_attack(dataset, args=None):
    dataset = generate_translation(dataset, args=args)
    return dataset

#     if self._use_google_translate and queue is None:
#         self._init_translate(translate_lang)

# def _init_translate(self, lang):
#     from_code = "en"
#     to_code = lang
#     argostranslate.package.update_package_index()
#     available_packages = argostranslate.package.get_available_packages()
#     package_to_install = next(
#         filter(
#             lambda x: (x.from_code == to_code and x.to_code == from_code),
#             available_packages,
#         )
#     )
#     argostranslate.package.install_from_path(package_to_install.download())
#     package_to_install = next(
#         filter(
#             lambda x: (x.from_code == from_code and x.to_code == to_code),
#             available_packages,
#         )
#     )
#     argostranslate.package.install_from_path(package_to_install.download())
    
    
# def _query_google_translate(
#     self, text: str, back_translate: bool = False
# ) -> str:
#     lang = "en" if back_translate else self._translate_lang
#     from_lang = self._translate_lang if back_translate else "en"

#     if self._queue is None:
#         answer = argostranslate.translate.translate(text, from_lang, lang)
#     else:
#         self._queue["translate"].put(
#             (text, from_lang, lang, self._resp_queue)
#         )
#         answer = self._resp_queue.get(block=True)

#     if not back_translate:
#         logger.debug("Text in %s:\n%s", self._translate_lang, answer)
#         answer = self._query_google_translate(answer, back_translate=True)
#     return answer
    

def check_output_column_lengths(example, min_len=0):
    baseline_completion_len = example["baseline_completion_length"]
    no_wm_output_len = example["no_wm_output_length"]
    w_wm_output_len = example["w_wm_output_length"]
    conds = all(
        [
            baseline_completion_len >= min_len,
            no_wm_output_len >= min_len,
            w_wm_output_len >= min_len,
        ]
    )
    return conds


def tokenize_for_copy_paste(example, tokenizer=None, args=None):
    for text_col in OUTPUT_TEXT_COLUMN_NAMES:
        if text_col in example:
            example[f"{text_col}_tokd"] = tokenizer(
                example[text_col], return_tensors="pt", add_special_tokens=False
            )["input_ids"][0].to(torch.int64)
    return example


def copy_paste_attack(example, tokenizer=None, args=None):
    # check if the example is long enough to attack
    if not check_output_column_lengths(example, min_len=args.cp_attack_min_len):
        # # if not, copy the orig w_wm_output to w_wm_output_attacked
        # NOTE changing this to return "" so that those fail/we can filter out these examples
        example["w_wm_output_attacked"] = ""
        example["w_wm_output_attacked_length"] = 0
        return example

    # else, attack

    # Understanding the functionality:
    # we always write the result into the "w_wm_output_attacked" column
    # however depending on the detection method we're targeting, the
    # "src" and "dst" columns will be different. However,
    # the internal logic for these functions has old naming conventions of
    # watermarked always being the insertion src and no_watermark always being the dst

    tokenized_dst = example[f"{args.cp_attack_dst_col}_tokd"]
    tokenized_src = example[f"{args.cp_attack_src_col}_tokd"]
    min_token_count = min(len(tokenized_dst), len(tokenized_src))

    if args.cp_attack_type == "single-single":  # 1-t
        if min_token_count < args.cp_attack_insertion_len:
            example["w_wm_output_attacked"] = example["w_wm_output"]
            example["w_wm_output_attacked_length"] = example["w_wm_output_length"]
            return example
        tokenized_attacked_output = single_insertion(
            args.cp_attack_insertion_len,
            min_token_count,
            tokenized_dst,
            tokenized_src,
        )
    elif args.cp_attack_type == "triple-single":  # 3-t
        if min_token_count < args.cp_attack_insertion_len:
            example["w_wm_output_attacked"] = example["w_wm_output"]
            example["w_wm_output_attacked_length"] = example["w_wm_output_length"]
            return example
        tokenized_attacked_output = triple_insertion_single_len(
            args.cp_attack_insertion_len,
            min_token_count,
            tokenized_dst,
            tokenized_src,
        )
    elif args.cp_attack_type == "k-t":
        tokenized_attacked_output = k_insertion_t_len(
            args.cp_attack_num_insertions,  # k
            args.cp_attack_insertion_len,  # t
            min_token_count,
            tokenized_dst,
            tokenized_src,
            verbose=args.verbose,
        )
    elif args.cp_attack_type == "k-random":  # k-t | k>=3, t in [floor(T/2k), T/k)
        raise NotImplementedError(f"Attack type {args.cp_attack_type} not implemented")
    elif args.cp_attack_type == "triple-triple":  # 3-(k_1,k_2,k_3)
        raise NotImplementedError(f"Attack type {args.cp_attack_type} not implemented")
    else:
        raise ValueError(f"Invalid attack type: {args.cp_attack_type}")

    example["w_wm_output_attacked"] = tokenizer.batch_decode(
        [tokenized_attacked_output.to(torch.int64)], skip_special_tokens=True
    )[0]
    example["w_wm_output_attacked_length"] = len(tokenized_attacked_output)

    return example


def get_helm_attack_att(attack_method):
    # attack = "helm_MisspellingAttack"
    att = getattr(helm, attack_method)()
    return att

def helm_attack(example, att=None, tokenizer=None, args=None):
    # check if the example is long enough to attack
    if not check_output_column_lengths(example, min_len=args.cp_attack_min_len):
        # # if not, copy the orig w_wm_output to w_wm_output_attacked
        # NOTE changing this to return "" so that those fail/we can filter out these examples
        example["w_wm_output_attacked"] = ""
        example["w_wm_output_attacked_length"] = 0
        return example
    
    w_wm_output_attacked = att.warp(example["w_wm_output"])
    example["w_wm_output_attacked"] = w_wm_output_attacked
    example["w_wm_output_attacked_length"] = len(tokenizer(w_wm_output_attacked)["input_ids"])
    return example
    
    
def get_swap_attack_att():
    # attack = "helm_MisspellingAttack"
    att = SwapAttack()
    return att

def swap_attack(example, att=None, tokenizer=None, args=None):
    # check if the example is long enough to attack
    if not check_output_column_lengths(example, min_len=args.cp_attack_min_len):
        # # if not, copy the orig w_wm_output to w_wm_output_attacked
        # NOTE changing this to return "" so that those fail/we can filter out these examples
        example["w_wm_output_attacked"] = ""
        example["w_wm_output_attacked_length"] = 0
        return example
    
    w_wm_output_attacked = att.warp(example["w_wm_output"])
    example["w_wm_output_attacked"] = w_wm_output_attacked
    example["w_wm_output_attacked_length"] = len(tokenizer(w_wm_output_attacked)["input_ids"])
    return example
    
    
def get_synonym_attack_att(args):
    # attack = "helm_MisspellingAttack"
    att = SynonymAttack(p=args.synonym_p)
    return att

def synonym_attack(example, att=None, tokenizer=None, args=None):
    # check if the example is long enough to attack
    if not check_output_column_lengths(example, min_len=args.cp_attack_min_len):
        # # if not, copy the orig w_wm_output to w_wm_output_attacked
        # NOTE changing this to return "" so that those fail/we can filter out these examples
        example["w_wm_output_attacked"] = ""
        example["w_wm_output_attacked_length"] = 0
        return example
    
    # tokenize_input = tokenizer(example["truncated_input"], return_tensors="pt")["input_ids"]
    w_wm_output_attacked = att.warp(example["w_wm_output"])
    example["w_wm_output_attacked"] = w_wm_output_attacked
    example["w_wm_output_attacked_length"] = len(tokenizer(w_wm_output_attacked)["input_ids"])
    return example
    
    
def translate_process(translation_queue, langs, device):
    import os
    import argostranslate.package
    import argostranslate.settings
    import argostranslate.translate

    # if device != "cpu":
    os.environ["ARGOS_DEVICE_TYPE"] = "cuda"
    argostranslate.settings.device = "cuda"
    # else:
    #     os.environ["ARGOS_DEVICE_TYPE"] = "cpu"
    #     argostranslate.settings.device = "cpu"

    # Install translation models
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()

    def install_model(la, lb):
        package_to_install = next(
            filter(
                lambda x: (x.from_code == la and x.to_code == lb),
                available_packages,
            )
        )
        argostranslate.package.install_from_path(package_to_install.download())

    for i, li in enumerate(langs):
        for j, lj in enumerate(langs):
            if i == j:
                continue
            try:
                install_model(li, lj)
            except Exception:
                pass

    # Get actual models
    pairs = {}

    while True:
        task = translation_queue.get(block=True)
        if task is None:
            return

        text, la, lb, dst_queue = task
        if (la, lb) not in pairs:
            pairs[
                (la, lb)
            ] = argostranslate.translate.get_translation_from_codes(la, lb)

        try:
            dst_queue.put(pairs[(la, lb)].translate(text))
        except RuntimeError as e:
            print(e)
            print("Reducing number of CUDA threads")
            translation_queue.put(task)
            return
              
def oracle_attack(dataset,args):
    oracle_args = oracle_att.get_cmd_args()
    # oracle_args.dataset = 'c4_realnews'
    attacker = oracle_att.Attacker()    

    
    # load more from the jsonl file...
    # prefix = f'{watermark_scheme}-watermark/{out_folder}' 
    # data = load_data(f"{prefix}/{dataset}_{watermark_scheme}.jsonl") 
    w_wm_output_attacked = []
    print(oracle_args)
    oracle = oracle_att.Oracle(check_quality=oracle_args.check_quality, choice_granuality=oracle_args.choice_granularity)
    
    for i, data in tqdm(enumerate(dataset), desc="Processing oracle phrases"):
        if "truncated_input" in list(data.keys()):
            query = data["truncated_input"]
        elif "question" in list(data.keys()):
            query = data["question"]
        else:
            raise ValueError("No query found in the data.")
        response = data["w_wm_output"]
        if data["w_wm_output_length"] < 30:
            w_wm_output_attacked.append(response)
            continue
        
        # datum = {"query": query,
        #         "output_with_watermark": response,
        #         }
        
            

        attacker.prefix = query
        oracle.set_query_and_response(query, response)
        # oracle = Oracle(query, response, check_quality=oracle_args.check_quality, choice_granuality=oracle_args.choice_granularity)
        print(f"Iteration {i}-th data:")
        print(f"Query: {query}")
        trainer = oracle_att.Trainer( oracle, oracle_args)
        result_dict = trainer.random_walk_attack(oracle, attacker)
        #Final output
        paraphrased_response = result_dict["paraphrased_response"]
        # print(f"Response: {response}")
        # print(f"Paraphrased Response: {paraphrased_response}")
        # result_dict["watermarked_response"] = datum["output_with_watermark"]
        # result_dict["query"] = query
    
        w_wm_output_attacked.append(paraphrased_response)
        
                
    dataset = dataset.add_column("w_wm_output_attacked", w_wm_output_attacked)
    return dataset
    
    