from __future__ import annotations
import collections
from math import sqrt

import scipy.stats

import torch
from torch import Tensor
from tokenizers import Tokenizer
from transformers import LogitsProcessor

from nltk.util import ngrams





# @misc{kuditipudi2023robust,
#       title={Robust Distortion-free Watermarks for Language Models}, 
#       author={Rohith Kuditipudi and John Thickstun and Tatsunori Hashimoto and Percy Liang},
#       year={2023},
#       eprint={2307.15593},
#       archivePrefix={arXiv},
#       primaryClass={cs.LG}
# }


import torch
from transformers import LogicProcessor  # Assuming LogicProcessor is a part of the transformers library
from mersenne import mersenne_rng

class rohith23_WatermarkGenerator(LogicProcessor):
    def __init__(self, model, vocab_size, n, m, key):
        super().__init__()  # Initialize the parent class
        self.model = model
        self.vocab_size = vocab_size
        self.n = n
        self.m = m
        self.key = key
        self.rng = mersenne_rng(self.key)  # Assuming mersenne_rng is defined elsewhere

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        xi = torch.tensor([self.rng.rand() for _ in range(self.n * self.vocab_size)]).view(self.n, self.vocab_size)
        shift = torch.randint(self.n, (1,))

    def generate_shift(self, tokens):
        xi = torch.tensor([self.rng.rand() for _ in range(self.n * self.vocab_size)]).view(self.n, self.vocab_size)
        shift = torch.randint(self.n, (1,))
        

        self.exp_sampling(scores, xi[(shift + i) % self.n, :]).to(self.model.device)
        
        inputs = tokens.to(self.model.device)
        attn = torch.ones_like(inputs)
        past = None
        for i in range(self.m):
            with torch.no_grad():
                if past:
                    output = self.model(inputs[:, -1:], past_key_values=past, attention_mask=attn)
                else:
                    output = self.model(inputs)

            probs = torch.nn.functional.softmax(output.logits[:, -1, :self.vocab_size], dim=-1).cpu()
            token = self.exp_sampling(probs, xi[(shift + i) % self.n, :]).to(self.model.device)
            inputs = torch.cat([inputs, token], dim=-1)

            past = output.past_key_values
            attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

        return inputs.detach().cpu()

    @staticmethod
    def exp_sampling(probs, u):
        return torch.argmax(u ** (1 / probs), axis=1).unsqueeze(-1)


