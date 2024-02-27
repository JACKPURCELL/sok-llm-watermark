import math
import random

import scipy
import torch
import torch.nn.functional as F

from .bit_tokenizer import Binarization
from .random import ExternalRandomness




class christ23_WatermarkLogitsProcessor:
    """
    A watermarking scheme that generates binary tokens. See Christ et al. (2023) for more details.

    Args:
        rng (RandomNumberGenerator): A random number generator.
        verifier (Verifier): A verifier object.
        tokenizer (Tokenizer): A tokenizer object.
        temp (float): A temperature value for softmax.
        binarizer (Binarization): A binarizer object.
        skip_prob (float): A probability value for skipping the watermarking process.

    Attributes:
        skip_prob (float): A probability value for skipping the watermarking process.
        base_len (int): The length of the previous tokens.
        binarizer (Binarization): A binarizer object.
    """

    def __init__(self, tokenizer, vocab_size, temp,  device,key=42, key_len=512, skip_prob=0.0):
        self.skip_prob = skip_prob
        self.binarizer = Binarization(tokenizer, device)
        self.base_len = -1
        self.rng = ExternalRandomness(key, device, len(tokenizer), key_len, self.binarizer.L)
        self.tokenizer = tokenizer
        self.temp = temp

    def reset(self):
        super().reset()
        self.base_len = -1

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # previous_tokens = self.tokenizer.decode(input_ids, return_tensors="pt")
        previous_tokens = input_ids
        previous_tokens = self.rng.normalize_previous_values(previous_tokens)

        # Truncate unused scores
        if random.random() < self.skip_prob:
            return scores

        scores = scores[:, : self.rng.vocab_size] / self.temp
        probs = F.softmax(scores, dim=-1)

        if self.base_len < 0:
            self.base_len = previous_tokens.shape[1]

        N, _ = scores.shape

        representations = self.binarizer.get_token_to_bits_tensor(probs.device)
        choice = torch.zeros((N, self.binarizer.L)).int().to(scores.device) - 1
        seeds = self.rng.get_seed(previous_tokens, input_ids)

        # Select next binary token
        branch_filter = torch.ones((N,)).to(probs.device).bool()
        for bit_index in range(self.binarizer.L):
            # If all choices are leaves, stop. We can tell we are on a leaf if the only remaining element has bit -1
            prob_sum = probs.sum(axis=1)
            prob_done = (
                probs[:, representations[:, bit_index] == -1].sum(axis=1)
                / prob_sum
            )
            branch_filter[prob_done > 0.5] = False
            if not branch_filter.sum():
                break

            # Compute probability and get randomness
            p = (
                probs[:, representations[:, bit_index] == 1].sum(axis=1)
                / prob_sum
            )
            h = self.rng.rand_index(seeds, bit_index, device=probs.device)

            choice[branch_filter, bit_index] = (h < p).int()[branch_filter]

            # Set probability of stale branches to 0
            criteria = (
                representations.expand(N, *representations.shape)[
                    :, :, bit_index
                ]
                != choice[:, bit_index].expand(self.binarizer.V, N).t()
            )
            probs[criteria] = 0

        # Convert to token
        try:
            choice = self.binarizer.to_token(choice)
        except Exception:
            # Sometimes (very rarely) the token does not exist. We need to debug this, but for now we just return the original scores.
            return scores

        next_token = choice.to(probs.device)
        scores[:] = -math.inf
        scores[torch.arange(scores.shape[0]), next_token] = 0

        return scores


class christ23_WatermarkDetector:
    """
    Verifier for binary watermarking schemes.

    Args:
        rng (RandomNumberGenerator): Random number generator.
        pvalue (float): P-value threshold for the statistical test.
        tokenizer (Tokenizer): Tokenizer object.
        binarizer (Binarizer): Binarizer object.
        skip_prob (float): Probability of skipping a token during verification.

    Attributes:
        skip_prob (float): Probability of skipping a token during verification.
        binarizer (Binarizer): Binarizer object.
    """

    def __init__(self, rng, pvalue, tokenizer, binarizer, skip_prob):

        self.pvalue = pvalue
        self.rng = ExternalRandomness(key, device, len(tokenizer), key_len, self.binarizer.L)
        self.tokenizer = tokenizer
        self.skip_prob = skip_prob
        if binarizer is None:
            self.binarizer = Binarization(tokenizer, rng.devices)
        else:
            self.binarizer = binarizer

    def detect(self, text, index=0, exact=False,**kwargs):
        output_dict = {}
        
        tokens = tokens.squeeze()
        if len(tokens.shape) == 0:
            return [(False, 0, 0, 0)]
        binary_tokens = self.binarizer.to_bit(
            tokens.to(self.rng.device)
        ).squeeze()
        mask = binary_tokens >= 0
        try:
            max_bitlen = mask.sum(axis=1).max()
        except Exception as exc:
            raise Exception("Error with max bitlen") from exc
        binary_tokens = binary_tokens[:, :max_bitlen]
        mask = mask[:, :max_bitlen]
        ctn = mask.sum(axis=1)

        xi = []

        for i in range(tokens.shape[-1]):
            prev_values = tokens[:i]
            bitlen = ctn[i].item()
            seed = self.rng.get_seed(prev_values, [index])
            xi.append(
                [self.rng.rand_index(seed, i).item() for i in range(bitlen)]
                + [-1 for _ in range(max_bitlen - bitlen)]
            )

        xi = torch.Tensor(xi).to(self.rng.device)

        v = (
            -(xi * binary_tokens + (1 - xi) * (1 - binary_tokens)).abs().log()
            * mask
        )
        cumul = v.sum(axis=-1).cumsum(0).tolist()
        ctn = mask.sum(axis=1).cumsum(0).tolist()

        # Compute average
        rtn = []
        for i, v in enumerate(cumul):
            c = ctn[i]
            likelihood = scipy.stats.gamma.sf(v, c)
            rtn.append((likelihood < self.pvalue, v - c, likelihood, i, i))

            output_dict["prediction"] = likelihood < self.pvalue
            output_dict["a"] = v - c
            output_dict["b"] = likelihood
            output_dict["c"] = c
            output_dict["i"] = i
        return output_dict

