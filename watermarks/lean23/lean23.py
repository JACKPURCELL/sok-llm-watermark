from __future__ import annotations
import torch
from transformers import LogitsProcessor, LogitsProcessorList, MinLengthLogitsProcessor
from message_model_processor import WmProcessorMessageModel
from lm_message_model import LMMessageModel
from watermark_processor import RepetitionPenaltyLogitsProcessor
import numpy as np


#@misc{wang2023codable,
#      title={Towards Codable Text Watermarking for Large Language Models},
#      author={Lean Wang and Wenkai Yang and Deli Chen and Hao Zhou and Yankai Lin and Fandong Meng and Jie Zhou and Xu Sun},
#      year={2023},
#      eprint={2307.15992},
#      archivePrefix={arXiv},
#      primaryClass={cs.CL}
#}


class lean23_BalanceMarkingWatermarkLogitsProcessor(LogitsProcessor):
    """
        Class for detecting watermarks in a sequence of tokens.

        Args:
            tokenizer: The tokenizer object for the main language model (in which we want to inject watermark).
            lm_model: The proxy model.
            lm_tokenizer: The tokenizer of the proxy model.

            strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
            vocab_size: The size of the vocabulary.
            watermark_key: The random seed for the green-listing.
    """
    def __init__(self,
                 tokenizer,
                 lm_tokenizer,
                 lm_model,
                 delta,
                 lm_prefix_len,
                 lm_top_k,
                 message_code_len,
                 random_permutation_num,
                 encode_ratio,
                 max_confidence_lbd,
                 message_model_strategy,
                 message,
                 top_k,
                 repeat_penalty):
        self.lm_message_model = LMMessageModel(tokenizer=tokenizer, lm_model=lm_model,
                                          lm_tokenizer=lm_tokenizer,
                                          delta=delta, lm_prefix_len=lm_prefix_len,
                                          lm_topk=lm_top_k, message_code_len=message_code_len,
                                          random_permutation_num=random_permutation_num)

        self.watermark_processor = WmProcessorMessageModel(message_model=self.lm_message_model,
                                                      tokenizer=tokenizer,
                                                      encode_ratio=encode_ratio,
                                                      max_confidence_lbd=max_confidence_lbd,
                                                      strategy=message_model_strategy,
                                                      message=message,
                                                      top_k=top_k,
                                                      )

        self.min_length_processor = MinLengthLogitsProcessor(min_length=10000,
                                                        # just to make sure there's no EOS
                                                        eos_token_id=tokenizer.eos_token_id)
        self.rep_processor = RepetitionPenaltyLogitsProcessor(penalty=repeat_penalty)

        self.logit_processor = (
            [self.min_length_processor, self.rep_processor, self.watermark_processor])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return self.logit_processor(input_ids, scores)


class lean23_WatermarkDetector:
    def __init__(self, watermarke_processor, generated_length, message_code_len, encode_ratio):
        self.watermark_processor = watermarke_processor
        self.generated_length = generated_length
        self.message_code_len = message_code_len
        self.encode_ratio = encode_ratio

    def detect(self, text):
        ### do we need this start_length
        #self.watermark_processor.start_length = tokenized_input['input_ids'].shape[-1]
        decoded_message, other_information = self.watermark_processor.decode(text, disable_tqdm=True)
        confidences = other_information[1]
        available_message_num = self.generated_length // (
            int(self.message_code_len * self.encode_ratio))
        acc = decoded_message[:available_message_num] == self.message[:available_message_num]
        result = {"decoded_message": decoded_message,
                  "confidences": confidences,
                  "accuracy": acc}
        return result






