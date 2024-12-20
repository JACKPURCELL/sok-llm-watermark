
import time
import torch
import transformers
import re
import math
import jsonlines
from tqdm import tqdm
import numpy as np
from .oracle import *

import argparse
import os

class oracle_Args:
    def __init__(self):
        self.dataset = "c4_realnews"
        self.cache_dir = "/data/jc/hf/hub"
        self.dist_alpha = 0.05
        self.checkpoint_alpha = 0.05
        self.check_quality = True
        self.watermark_scheme = "umd"
        self.choice_granularity = 5
        self.oracle_model = "gpt-3.5"
        self.tie_threshold = 0.001
        self.repetition_penalty = 1.1
        self.mask_top_p = 0.95
        self.n_spans = 1
        self.span_len = 6
        self.step_T = 400
        self.mask_filling_model_name = "google/t5-v1_1-xl"
        self.chunk_size = 20
        self.int8 = False
        self.half = False
        self.buffer_size = 1

def get_cmd_args():
    return oracle_Args()

class Attacker:
    def __init__(self) -> None:
        self.n_resample = 5
        self.oracle_args = get_cmd_args()
        self.mask_filling_model_name = self.oracle_args.mask_filling_model_name
        self.n_positions = 512 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if 't5' in self.oracle_args.mask_filling_model_name:
            self.mask_model = self.load_mask_model()
        self.mask_tokenizer = transformers.AutoTokenizer.from_pretrained(self.oracle_args.mask_filling_model_name, model_max_length=self.n_positions, cache_dir=self.oracle_args.cache_dir)
        self.query = None
        self.response = None
        self.verbose = True
        self.init_score = None
        self.start_idx = 4
        self.prefix = ""
        
        self.cached_replaced_tokens = set()
        self.original_tokens = set()

    def load_mask_model(self):
        int8_kwargs = {}
        half_kwargs = {}
        if self.oracle_args.int8:
            int8_kwargs = dict(load_in_8bit=True, device_map='auto', torch_dtype=torch.bfloat16)
        elif self.oracle_args.half:
            half_kwargs = dict(torch_dtype=torch.bfloat16)
        print(f'Loading mask filling model {self.oracle_args.mask_filling_model_name}...')
        mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(self.oracle_args.mask_filling_model_name, **int8_kwargs, **half_kwargs)
        try:
            self.n_positions = self.mask_model.config.n_positions
        except AttributeError:
            self.n_positions = 512
        print('MOVING MASK MODEL TO GPU...', end='', flush=True)
        start = time.time()
        # if not self.oracle_args.random_fills and not self.oracle_args.int8:
        mask_model.to(self.device)
        print(f'DONE ({time.time() - start:.2f}s)')
        return mask_model
    
    def tokenize_and_mask(self, text, span_len, pct, ceil_pct=False):
        tokens = text.replace('\n', ' \n').split(' ')
        mask_string = '<<<mask>>>'
        # only mask one span
        n_spans = self.oracle_args.n_spans

        if ceil_pct:
            n_spans = np.ceil(n_spans)
        n_spans = int(n_spans)

        n_masks = 0
        while n_masks < n_spans:
            start_pos = 0 # only need to prevent moefiying the instruction for chat models as they repeat Q-A.
            # if self.oracle_args.dataset == "c4_realnews":
            #     start_pos = 0
            # else:
            # # chat models might repeat the Q:.... A: prompt. So avoid query being perturbed.
            #     start_pos = len(self.prefix.replace('\n', ' \n').split(' ')) 
            start = np.random.randint(start_pos, len(tokens) - span_len)

            end = start + span_len
            search_start = max(0, start - self.oracle_args.buffer_size)
            search_end = min(len(tokens), end + self.oracle_args.buffer_size)
            if mask_string not in tokens[search_start:search_end]:
                # record/remove already masked tokens
                masked_tokens = set(tokens[start:end])
                if len(masked_tokens) > 1:
                    self.cached_replaced_tokens |= masked_tokens
                tokens[start:end] = [mask_string]
                n_masks += 1
        
        num_filled = 0
        for idx, token in enumerate(tokens):
            if token == mask_string:
                tokens[idx] = f'<extra_id_{num_filled}>'
                num_filled += 1
        assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
        text = ' '.join(tokens)
        return text

    def count_masks(self, texts):
        return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]
    
    # replace each masked span with a sample from T5 mask_model
    def replace_masks(self, texts):
        n_expected = count_masks(texts)
        stop_id = self.mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
        tokens = self.mask_tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        
        min_len = int(np.ceil(self.oracle_args.span_len * self.oracle_args.n_spans * 1.5)) 
        max_len = int(self.oracle_args.span_len*self.oracle_args.n_spans*2)
        print("min length: ", min_len, "max length: ", max_len)
        outputs = self.mask_model.generate(**tokens, max_length=max_len, min_length=min_len, do_sample=True, top_p=self.oracle_args.mask_top_p, num_return_sequences=1, repetition_penalty=self.oracle_args.repetition_penalty, eos_token_id=stop_id)  # 500 max, 150
        return self.mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)

    def perturb_texts_(self, texts, span_len, pct, ceil_pct=False):
        masked_texts = []
        for x in texts:
            masked_texts.append(self.tokenize_and_mask(x, span_len, pct, ceil_pct))

        raw_fills = self.replace_masks(masked_texts)
        extracted_fills = extract_fills(raw_fills)
        perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

        # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
        attempts = 1
        while '' in perturbed_texts:
            idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
            print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
            masked_texts = [self.tokenize_and_mask(x, span_len, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
            raw_fills = self.replace_masks(masked_texts)
            extracted_fills = extract_fills(raw_fills)
            new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
            for idx, x in zip(idxs, new_perturbed_texts):
                perturbed_texts[idx] = x
            attempts += 1
        
        return perturbed_texts

    def perturb_texts_t5(self, texts, span_len, pct, k=5, ceil_pct=False):
      chunk_size = self.oracle_args.chunk_size
      if '11b' in self.oracle_args.mask_filling_model_name:
          chunk_size //= 2

      outputs = []
      # set chunk_size as 1 to help make sure each original token is replaced.
      for i in tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
          outputs.extend(self.perturb_texts_(texts[i:i + chunk_size], span_len, pct, ceil_pct=ceil_pct))
      return outputs

    def paraphrase(self, texts, k=5):
        return self.perturb_texts_t5(texts, span_len=self.oracle_args.span_len, pct=0.2, k=k, ceil_pct=False)

class Trainer():
    def __init__(self,  oracle, verbose=True):
        self.oracle = oracle
        self.verbose = verbose
        self.n_resample = 100
        self.steps = 10
        self.oracle_args = get_cmd_args()
        self.mask_filling_model_name = self.oracle_args.mask_filling_model_name
        self.n_positions = 512 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # def load_mask_model(self):
    #     # mask filling t5 model
    #     int8_kwargs = {}
    #     half_kwargs = {}
    #     if self.oracle_args.int8:
    #         int8_kwargs = dict(load_in_8bit=True, device_map='auto', torch_dtype=torch.bfloat16)
    #     elif self.oracle_args.half:
    #         half_kwargs = dict(torch_dtype=torch.bfloat16)
    #     print(f'Loading mask filling model {self.mask_filling_model_name}...')
    #     mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(self.mask_filling_model_name, **int8_kwargs, **half_kwargs, cache_dir=self.oracle_args.cache_dir)
    #     try:
    #         self.n_positions = self.mask_model.config.n_positions
    #     except AttributeError:
    #         self.n_positions = 512
    #     print('MOVING MASK MODEL TO GPU...', end='', flush=True)
    #     start = time.time()
    #     mask_model.to(self.device)
    #     print(f'DONE ({time.time() - start:.2f}s)')
    #     return mask_model
    
    def random_walk_attack(self, oracle, attacker):
        # find a perturbation (through repeated sampling) such that the quality oracle says quality is maintained; then repeat, several times 
        response = oracle.response
        dist = -1
        n_iter, max_rnd_steps = 0, 200
        rnd_walk_step = 0
        # set the stopping criterion
        threshold_dist = self.oracle_args.dist_alpha * len(oracle.response)
        attacker.original_tokens = set(response.replace("\n", " ").split(" ")) 
        threshold_dist = int(self.oracle_args.dist_alpha * len(attacker.original_tokens))
        checkpoint_dist = int(self.oracle_args.checkpoint_alpha * len(attacker.original_tokens))
        maintain_quality_or_not = True
        patience = 0
        ckpt_cnt = 0
        mixing_patience = 0
        intermediate_examples = [response]
        while n_iter < self.oracle_args.step_T:
            last_replaced_tokens = set()
            attacker.cached_replaced_tokens = set()
            score = 0
            n_response = attacker.paraphrase([response], self.oracle_args.span_len)[0]
            n_response = re.sub(r'\s{2,}', ' ', n_response) # strip the extra spaces
            if oracle.maintain_quality(n_response, model=self.oracle_args.oracle_model, tie_threshold=self.oracle_args.tie_threshold):
                response = n_response
                rnd_walk_step += 1
                attacker.original_tokens -= attacker.cached_replaced_tokens
                last_replaced_tokens = attacker.cached_replaced_tokens
                patience = 0
                if int(rnd_walk_step*self.oracle_args.span_len) // checkpoint_dist > ckpt_cnt:
                    intermediate_examples.append(n_response)
                    ckpt_cnt += 1
                print("Get a better response.")
            n_iter += 1
            # dist = Levenshtein.distance(response, oracle.response)
            if n_iter % 10 == 0:
                print("Original Text: ")
                print(oracle.response.__repr__())
            print(f"Walk {rnd_walk_step} / Iteration {n_iter}, {len(attacker.original_tokens)} > {threshold_dist} unique tokens replaced, Paraphrased Text:")
            print(n_response.__repr__())
            if rnd_walk_step >= max_rnd_steps or patience >= 150:
                print("Max random walk steps reached. Exiting.")
                break 
            if len(attacker.original_tokens) <= threshold_dist:
                mixing_patience += 1
            if mixing_patience > self.oracle_args.step_T/3:
                print("Mixing patience exceeded. Exiting.")
                break
            if patience > 30:
                print("Remaining tokens to be masked.")
                print(attacker.original_tokens)
                print("Patience exceeded. Backtrack.")
                response = intermediate_examples[-1]
                attacker.original_tokens = last_replaced_tokens | attacker.original_tokens
            patience += 1

        if patience >= 150:
            maintain_quality_or_not = False
        if self.verbose:
            print("Step: ", n_iter)
            print("Original Text: ")
            print(oracle.response.__repr__())
            print("Paraphrase: ")
            print(response.__repr__())
            print("Quality: ")
            print(score)
            print(f"Quality maintained: {maintain_quality_or_not}")
            print()
            
        result_dict = {"watermarked_response": oracle.response, "paraphrased_response": response, "maintain_quality_or_not": maintain_quality_or_not, "patience": patience}
        if len(intermediate_examples) > 1: # intermediate steps for checkpointing etc
            result_dict["intermediate_examples"] = intermediate_examples

        return result_dict

def run_once(query, response=None):
    oracle_args = get_cmd_args()
    oracle_args.dataset == 'lfqa'
    attacker = Attacker()
    result_dict = {}
    result_dict["watermarked_response"] = response

    attack_results = []
    oracle = Oracle(query, response, check_quality=oracle_args.check_quality, choice_granuality=oracle_args.choice_granularity, cache_dir=oracle_args.cache_dir)
    print(f"Query: {query}")
    data = None
    trainer = Trainer(data, oracle, oracle_args)
    result_dict = trainer.random_walk_attack(oracle, attacker)
    paraphrased_response = result_dict["paraphrased_response"]
    print(f"Response: {response}")
    print(f"Paraphrased Response: {paraphrased_response}")
    # result_dict["answer"] = answer
    result_dict["query"] = query
    attack_results.append(result_dict)
    print("Final results:")
    print(attack_results)



def oracle_attack(dataset,args):
    oracle_args = get_cmd_args()
    # oracle_args.dataset = 'c4_realnews'
    attacker = Attacker()    

    
    # load more from the jsonl file...
    # prefix = f'{watermark_scheme}-watermark/{out_folder}' 
    # data = load_data(f"{prefix}/{dataset}_{watermark_scheme}.jsonl") 
    w_wm_output_attacked = []
    print(oracle_args)
    oracle = Oracle(check_quality=oracle_args.check_quality, choice_granuality=oracle_args.choice_granularity)
    
    for i, data in tqdm(enumerate(dataset), desc="Processing oracle phrases"):
        if "truncated_input" in list(data.keys()):
            query = data["truncated_input"]
        elif "question" in list(data.keys()):
            query = data["question"]
        else:
            raise ValueError("No query found in the data.")
        response = data["w_wm_output"]
        
        datum = {"query": query,
                "output_with_watermark": response,
                }
        
            

        if "prefix" in list(datum.keys()):
            query = datum["prefix"]
        elif "query" in list(datum.keys()):
            query = datum["query"]
        else:
            query = None
        
        attacker.prefix = query
        oracle.set_query_and_response(query, response)
        # oracle = Oracle(query, response, check_quality=oracle_args.check_quality, choice_granuality=oracle_args.choice_granularity)
        print(f"Iteration {i}-th data:")
        print(f"Query: {query}")
        trainer = Trainer(data, oracle, oracle_args)
        result_dict = trainer.random_walk_attack(oracle, attacker)
        #Final output
        paraphrased_response = result_dict["paraphrased_response"]
        # print(f"Response: {response}")
        # print(f"Paraphrased Response: {paraphrased_response}")
        # result_dict["watermarked_response"] = datum["output_with_watermark"]
        # result_dict["query"] = query
    
        w_wm_output_attacked.append(paraphrased_response)
        
                
    data = data.add_column("w_wm_output_attacked", w_wm_output_attacked)
    


if __name__ == '__main__':
    oracle_args = get_cmd_args()
    print(oracle_args)
    main(query=oracle_args.query,
         response=oracle_args.response)