# @article{zhao2023provable,
#   title={Provable Robust Watermarking for AI-Generated Text},
#   author={Zhao, Xuandong and Ananth, Prabhanjan and Li, Lei and Wang, Yu-Xiang},
#   journal={arXiv preprint arXiv:2306.17439},
#   year={2023}
# }
# https://arxiv.org/abs/2306.17439


import hashlib
from typing import List
import numpy as np
from scipy.stats import norm
import torch
from transformers import LogitsWarper


class GPTWatermarkBase:
    """
    Base class for watermarking distributions with fixed-group green-listed tokens.

    Args:
        fraction: The fraction of the distribution to be green-listed.
        strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
        vocab_size: The size of the vocabulary.
        watermark_key: The random seed for the green-listing.
    """

    def __init__(self, fraction: float = 0.5, tokenizer=None, strength: float = 2.0, vocab_size: int = 50257, watermark_key: int = 0, threshold:float=4.0,**kwargs):
        rng = np.random.default_rng(self._hash_fn(watermark_key))
        mask = np.array([True] * int(fraction * vocab_size) + [False] * (vocab_size - int(fraction * vocab_size)))
        rng.shuffle(mask)
        self.green_list_mask = torch.tensor(mask, dtype=torch.float32)
        self.strength = strength
        self.fraction = fraction
        self.min_prefix_len = 1
        self.tokenizer = tokenizer
        self.threshold = threshold
    @staticmethod
    def _hash_fn(x: int) -> int:
        """solution from https://stackoverflow.com/questions/67219691/python-hash-function-that-returns-32-or-64-bits"""
        x = np.int64(x)
        return int.from_bytes(hashlib.sha256(x).digest()[:4], 'little')


class xuandong23b_WatermarkLogitsProcessor(GPTWatermarkBase, LogitsWarper):
    """
    LogitsWarper for watermarking distributions with fixed-group green-listed tokens.

    Args:
        fraction: The fraction of the distribution to be green-listed.
        strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
        vocab_size: The size of the vocabulary.
        watermark_key: The random seed for the green-listing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.FloatTensor:
        """Add the watermark to the logits and return new logits."""
        watermark = self.strength * self.green_list_mask
        new_logits = scores + watermark.to(scores.device)
        return new_logits


class xuandong23b_WatermarkDetector(GPTWatermarkBase):
    """
    Class for detecting watermarks in a sequence of tokens.

    Args:
        fraction: The fraction of the distribution to be green-listed.
        strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
        vocab_size: The size of the vocabulary.
        watermark_key: The random seed for the green-listing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _z_score(num_green: int, total: int, fraction: float) -> float:
        """Calculate and return the z-score of the number of green tokens in a sequence."""
        return (num_green - fraction * total) / np.sqrt(fraction * (1 - fraction) * total)
    
    @staticmethod
    def _compute_tau(m: int, N: int, alpha: float) -> float:
        """
        Compute the threshold tau for the dynamic thresholding.

        Args:
            m: The number of unique tokens in the sequence.
            N: Vocabulary size.
            alpha: The false positive rate to control.
        Returns:
            The threshold tau.
        """
        factor = np.sqrt(1 - (m - 1) / (N - 1))
        tau = factor * norm.ppf(1 - alpha)
        return tau

    def detect(self, text,**kwargs) -> float:
        """Detect the watermark in a sequence of tokens and return the z value."""
        tokenized_text = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        green_tokens = int(sum(self.green_list_mask[i] for i in tokenized_text))
        output_dict = {}
        output_dict["z_score"] = self._z_score(green_tokens, len(tokenized_text), self.fraction)        
        output_dict["prediction"] = bool(output_dict["z_score"] > self.threshold)
        return output_dict
    

    def dummy_detect(self,  **kwargs):
        result = {
                  "z_score": float("nan"),
                  "prediction": False}

        return result


    
    def unidetect(self, sequence: List[int]) -> float:
        """Detect the watermark in a sequence of tokens and return the z value. Just for unique tokens."""
        sequence = list(set(sequence))
        green_tokens = int(sum(self.green_list_mask[i] for i in sequence))
        return self._z_score(green_tokens, len(sequence), self.fraction)
    
    def dynamic_threshold(self, sequence: List[int], alpha: float, vocab_size: int) -> (bool, float):
        """Dynamic thresholding for watermark detection. True if the sequence is watermarked, False otherwise."""
        z_score = self.unidetect(sequence)
        tau = self._compute_tau(len(list(set(sequence))), vocab_size, alpha)
        output_dict = {}
        output_dict["z_score"] = z_score        
        output_dict["prediction"] = z_score > tau
        return output_dict