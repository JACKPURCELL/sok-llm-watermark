from __future__ import annotations
import torch
from tqdm import tqdm
from transformers import LogitsProcessor
from .mersenne import mersenne_rng
import numpy as np

import os, sys, argparse, time
import pyximport
pyximport.install(reload_support=True, language_level=sys.version_info[0],
                  setup_args={'include_dirs':np.get_include()})
from .levenshtein import levenshtein

from scipy.stats import norm
from datasets import load_dataset
import pickle
# from math import sqrt, log

# def levenshtein(x, y, gamma=0.0):
#     n = len(x)
#     m = len(y)

#     A = np.zeros((n+1, m+1), dtype=np.float32)

#     for i in range(0, n+1):
#         for j in range(0, m+1):
#             if i == 0:
#                 A[i][j] = j * gamma
#             elif j == 0:
#                 A[i][j] = i * gamma
#             else:
#                 cost = log(1 - y[j-1, x[i-1]])
#                 A[i][j] = A[i-1][j] + gamma
#                 if A[i][j-1] + gamma < A[i][j]:
#                     A[i][j] = A[i][j-1] + gamma
#                 if A[i-1][j-1] + cost < A[i][j]:
#                     A[i][j] = A[i-1][j-1] + cost

#     return A[n][m]


# @misc{kuditipudi2023robust,
#       title={Robust Distortion-free Watermarks for Language Models},
#       author={Rohith Kuditipudi and John Thickstun and Tatsunori Hashimoto and Percy Liang},
#       year={2023},
#       eprint={2307.15593},
#       archivePrefix={arXiv},
#       primaryClass={cs.LG}
# }

    # parser.add_argument('--model',default='facebook/opt-1.3b',type=str,
    #         help='a HuggingFace model id of the model to generate from')
    # parser.add_argument('--prompt',default='',type=str,
    #         help='an optional prompt for generation')
    # parser.add_argument('--m',default=80,type=int,
    #         help='the requested length of the generated text')
    # parser.add_argument('--n',default=256,type=int,
    #         help='the length of the watermark sequence')
    # parser.add_argument('--key',default=42,type=int,
    #         help='a key for generating the random watermark sequence')
    # parser.add_argument('--seed',default=0,type=int,
    #         help='a seed for reproducibile randomness')
    
    
# class WatermarkBase:
#     """
#     Base class for watermarking distributions with fixed-group green-listed tokens.

#     Args:
#         fraction: The fraction of the distribution to be green-listed.
#         strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
#         vocab_size: The size of the vocabulary.
#         watermark_key: The random seed for the green-listing.
#     """

#     def __init__(self, vocab_size, n=256, key=42):

#         self.n = n
#         self.rng = mersenne_rng(key)
#         self.vocab_size = vocab_size
        
class rohith23_WatermarkLogitsProcessor(LogitsProcessor):
    def __init__(self, vocab_size, n=50, key=42):
        self.n = n

        self.rng = np.random.default_rng(key)
        
        self.vocab_size = vocab_size
        self.shift = torch.randint(self.n, (1,))
        self.xi = torch.from_numpy(self.rng.random((self.n, self.vocab_size)).astype(np.float32)).cuda()
        # torch.tensor([self.rng.rand() for _ in range(self.n * self.vocab_size)]).view(self.n, self.vocab_size)
        self.i = 0

        
        ### If we use the same logits processor, this i will be cumulatively added

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        probs = torch.nn.functional.softmax(scores, dim=-1) #scores = logits?
        u = self.xi[(self.shift + self.i) % self.n, :]
        self.i += 1
        return u ** (1 / probs)


### If it is okay that we can have no classes but only funcitons, just put these two funcitons out
class rohith23_WatermarkDetector:
    def __init__(self,  vocab_size, tokenizer, args,n=50, key=42, fast=True):
        self.n = n
        self.rng = np.random.default_rng(key)
        self.vocab_size = vocab_size
        self.min_prefix_len = 1
        self.tokenizer=tokenizer
        self.xi = self.rng.random((self.n, self.vocab_size)).astype(np.float32)
        self.args = args
        if fast:
            self.build_null_results()

        # pval = permutation_test(tokens,args.key,args.n,len(tokens),len(tokenizer))
    
    def build_null_results(self, n_runs=1000):
        path = "./watermark_reliability_release/watermarks/rohith23/null_results_1000"+self.args.max_new_tokens+self.model_name_or_path+".pkl"
        if os.path.exists(path):
            print("Null results already exist, loading.")
            self.null_results = pickle.load(open(path, "rb"))
            print("Loading finished.")
        else:
            print("Building null results.")
            ### dataset change here
            dataset = load_dataset("c4", "realnewslike", split="train", streaming=True)
            ds_iterator = iter(dataset)
            null_results = []
            pbar = tqdm(total=n_runs)
            run = 0
            while run < n_runs:
                example = next(ds_iterator)
                text = example['text']
                tokenized_text = self.tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=2048).numpy()[0]
                if tokenized_text[0] == self.tokenizer.bos_token_id:
                    tokenized_text = tokenized_text[1:]
                tokenized_text = tokenized_text[:100]
                k = len(tokenized_text)
                '''tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=2048-buffer_tokens)[0]
                if len(tokens) < prompt_tokens + new_tokens:
                    continue
                tokens = tokens[-new_tokens:]

                seed = torch.randint(high=args.max_seed,size=(1,)).item()
                generator.manual_seed(int(seed))'''
                xi = np.random.rand(self.n, self.vocab_size).astype(np.float32)
                null_result = self._detect(tokenized_text, k, xi)
                null_results.append(null_result)
                run += 1
                pbar.update(1)

            self.null_results = torch.sort(torch.tensor(null_results)).values
            pickle.dump(self.null_results, open(path, "wb"))
        
    
    @staticmethod
    def _detect(tokens, k, xi, gamma=0.0):
        m = len(tokens)
        n = len(xi)
                
        A = np.empty((m - (k - 1), n))
        for i in range(m - (k - 1)):
            for j in range(n):
                A[i][j] = levenshtein(tokens[i:i + k], xi[(j + np.arange(k)) % n], gamma)

        return np.min(A)

    def slow_detect(self,text,n_runs=100,**kwargs):
        # tokenized_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].cuda()
        tokenized_text = self.tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=2048).numpy()[0]
        if tokenized_text[0] == self.tokenizer.bos_token_id:
            tokenized_text = tokenized_text[1:]
            
        output_dict = {}
        
        k = len(tokenized_text)
        output_dict["num_tokens_scored"] = k
        
        # xi = self.rng.random((self.n, self.vocab_size)).astype(np.float32)
        
        test_result = self._detect(tokenized_text,  k, self.xi)

        p_val = 0
        for i in range(n_runs):
            xi_alternative = np.random.rand(self.n, self.vocab_size).astype(np.float32)
            null_result = self._detect(tokenized_text, k, xi_alternative)
            # assuming lower test values indicate presence of watermark
            p_val += null_result <= test_result
        output_dict["p_val"] = p_val
        output_dict["p-value"] = (p_val + 1.0) / (n_runs + 1.0)
        output_dict["z_score"] = norm.ppf(1 - output_dict["p-value"])
        output_dict["prediction"] = output_dict["p_val"] < 0.02
        return output_dict
    
    def detect(self, text, n_runs=1000, **kwargs):
        # tokenized_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].cuda()
        tokenized_text = self.tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=2048).numpy()[0]
        if tokenized_text[0] == self.tokenizer.bos_token_id:
            tokenized_text = tokenized_text[1:]
            
        output_dict = {}
        
        k = len(tokenized_text)
        output_dict["num_tokens_scored"] = k
        
        test_result = self._detect(tokenized_text,  k, self.xi)

        p_val = torch.searchsorted(self.null_results, test_result, right=True) 
        '''p_val = 0
        for i in range(n_runs):
            # assuming lower test values indicate presence of watermark
            p_val += self.null_results[i] <= test_result'''

        output_dict["p_val"] = p_val
        output_dict["p-value"] = (p_val + 1.0) / (n_runs + 1.0)
        output_dict["z_score"] = norm.ppf(1 - output_dict["p-value"])
        output_dict["prediction"] = output_dict["p_val"] < 0.02
        return output_dict

      
    def dummy_detect(self, **kwargs):
        result = {"p_val": float("nan"),
                  "z_score": float("nan"),
                    "p-value": float("nan"),
                  "num_tokens_scored":float("nan"),
                  "prediction": False}

        return result


       


