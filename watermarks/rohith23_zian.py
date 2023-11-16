from __future__ import annotations
import collections
from math import sqrt

import scipy.stats

import torch
from torch import Tensor
from tokenizers import Tokenizer
from transformers import LogitsProcessor, AutoTokenizer, AutoModelForCausalLM
from mersenne import mersenne_rng
from nltk.util import ngrams





# @misc{kuditipudi2023robust,
#       title={Robust Distortion-free Watermarks for Language Models},
#       author={Rohith Kuditipudi and John Thickstun and Tatsunori Hashimoto and Percy Liang},
#       year={2023},
#       eprint={2307.15593},
#       archivePrefix={arXiv},
#       primaryClass={cs.LG}
# }


class rohith23_WatermarkLogitsProcessor(LogitsProcessor):
    def __int__(self, vocab_size, prompt, m, n, key, seed):
        self.m = m  # seems not useful?
        self.n = n
        self.shift = torch.randint(n, (1,))
        ### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.vocab_size = vocab_size
        rng = mersenne_rng(key)
        self.xi = torch.tensor([rng.rand() for _ in range(n * self.vocab_size)]).view(n, self.vocab_size)
        torch.manual_seed(seed) ### works?

        ### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.i = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        probs = torch.nn.functional.softmax(scores[:, -1, :self.vocab_size], dim=-1).cpu() #scores = logits?
        u = self.xi[(self.shift + self.i) % self.n, :]
        self.i += 1
        return u ** (1 / probs)





