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


class rohith23_WatermarkLogitsProcessor(LogitsProcessor):
    def __int__(self, vocab_size, prompt, m, n, key, seed):
        torch.manual_seed(seed)  ### works?
        self.m = m  # seems not useful?
        self.n = n
        self.shift = torch.randint(n, (1,))
        ### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.vocab_size = vocab_size
        rng = mersenne_rng(key)
        self.xi = torch.tensor([rng.rand() for _ in range(n * self.vocab_size)]).view(n, self.vocab_size)

        ### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.i = 0
        ### If we use the same logits processor, this i will be cumulatively added

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        probs = torch.nn.functional.softmax(scores[:, -1, :self.vocab_size], dim=-1).cpu() #scores = logits?
        u = self.xi[(self.shift + self.i) % self.n, :]
        self.i += 1
        return u ** (1 / probs)


### If it is okay that we can have no classes but only funcitons, just put these two funcitons out
class rohith23_WatermarkDetector():
    def __init__(self):
        pass
    @staticmethod
    def permutation_test(tokens, vocab_size, n, key, n_runs=100):
        rng = mersenne_rng(key)
        k = len(tokens)
        xi = np.array([rng.rand() for _ in range(n * vocab_size)], dtype=np.float32).reshape(n, vocab_size)
        test_result = rohith23_WatermarkDetector.detect(tokens, n, k, xi)

        p_val = 0
        for run in range(n_runs):
            xi_alternative = np.random.rand(n, vocab_size).astype(np.float32)
            null_result = rohith23_WatermarkDetector.detect(tokens, n, k, xi_alternative)

            # assuming lower test values indicate presence of watermark
            p_val += null_result <= test_result

        return (p_val + 1.0) / (n_runs + 1.0)

    @staticmethod
    def detect(tokens, k, xi, gamma=0.0):
        m = len(tokens)
        n = len(xi)

        A = np.empty((m - (k - 1), n))
        for i in range(m - (k - 1)):
            for j in range(n):
                A[i][j] = levenshtein(tokens[i:i + k], xi[(j + np.arange(k)) % n], gamma)

        return np.min(A)






