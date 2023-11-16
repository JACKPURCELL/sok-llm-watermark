from __future__ import annotations
import torch
from transformers import LogitsProcessor
from mersenne import mersenne_rng
import numpy as np
from levenshtein import levenshtein





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
    
    
class WatermarkBase:
    """
    Base class for watermarking distributions with fixed-group green-listed tokens.

    Args:
        fraction: The fraction of the distribution to be green-listed.
        strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
        vocab_size: The size of the vocabulary.
        watermark_key: The random seed for the green-listing.
    """

    def __init__(self, vocab_size, n=256, key=42):

        self.n = n
        self.rng = mersenne_rng(key)
        self.vocab_size = vocab_size
        
class rohith23_WatermarkLogitsProcessor(LogitsProcessor,WatermarkBase):
    def __int__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.shift = torch.randint(n, (1,))
        self.xi = torch.tensor([self.rng.rand() for _ in range(n * self.vocab_size)]).view(n, self.vocab_size)
        self.i = 0
        ### If we use the same logits processor, this i will be cumulatively added

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        probs = torch.nn.functional.softmax(scores[:, -1, :self.vocab_size], dim=-1).cpu() #scores = logits?
        u = self.xi[(self.shift + self.i) % self.n, :]
        self.i += 1
        return u ** (1 / probs)


### If it is okay that we can have no classes but only funcitons, just put these two funcitons out
class rohith23_WatermarkDetector(WatermarkBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # pval = permutation_test(tokens,args.key,args.n,len(tokens),len(tokenizer))
    
    @staticmethod
    def _detect(tokens, k, xi, gamma=0.0):
        m = len(tokens)
        n = len(xi)

        A = np.empty((m - (k - 1), n))
        for i in range(m - (k - 1)):
            for j in range(n):
                A[i][j] = levenshtein(tokens[i:i + k], xi[(j + np.arange(k)) % n], gamma)

        return np.min(A)

    def detect(self,tokens, n_runs=100):
        k = len(tokens)
        xi = np.array([self.rng.rand() for _ in range(self.n * self.vocab_size)], dtype=np.float32).reshape(self.n, self.vocab_size)
        test_result = self._detect(tokens, self.n, k, xi)

        p_val = 0
        for i in range(k):
            xi_alternative = np.random.rand(self.n, self.vocab_size).astype(np.float32)
            null_result = self._detect(tokens, k, xi_alternative)

            # assuming lower test values indicate presence of watermark
            p_val += null_result <= test_result

        return (p_val + 1.0) / (k + 1.0)








