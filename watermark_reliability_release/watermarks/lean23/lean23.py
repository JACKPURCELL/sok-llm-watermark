import torch
from .watermark_processor import RepetitionPenaltyLogitsProcessor
from transformers import LogitsProcessor, LogitsProcessorList, MinLengthLogitsProcessor
from .watermark_processors.message_model_processor import WmProcessorMessageModel
from .watermark_processors.message_models.lm_message_model import LMMessageModel


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

        self.logit_processor = LogitsProcessorList(
            [self.min_length_processor, self.rep_processor, self.watermark_processor])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return self.logit_processor(input_ids, scores)

def truncate(d, max_length=200):
    for k, v in d.items():
        if isinstance(v, torch.Tensor) and len(v.shape) == 2:
            d[k] = v[:, :max_length]
    return d
class lean23_WatermarkDetector:
    def __init__(self,
                 watermark_processor,
                 generated_length,
                 message_code_len,
                 encode_ratio,
                 tokenizer,
                 prompt_length,
                 message,
                 min_prefix_len=0
                 ):
        self.watermark_processor = watermark_processor
        self.generated_length = generated_length
        self.message_code_len = message_code_len
        self.encode_ratio = encode_ratio
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.min_prefix_len = min_prefix_len
        self.message = message

    def detect(self, text, prompt, **kwargs):
        if len(self.tokenizer.tokenize(text))<12:
            return {"decoded_message": "",
                  "confidences": 1,
                  "prediction": False}
        tokenized_input = self.tokenizer(prompt, return_tensors='pt').to("cpu")
        tokenized_input = truncate(tokenized_input, max_length=self.prompt_length)
        self.watermark_processor.start_length = tokenized_input['input_ids'].shape[-1]
        decoded_message, other_information = self.watermark_processor.watermark_processor.decode(text, disable_tqdm=True)
        confidences = other_information[1]
        available_message_num = self.generated_length // (
            int(self.message_code_len * self.encode_ratio))
        acc = decoded_message[:available_message_num] == self.message[:available_message_num]
        result = {
                  "confidences": confidences,
                  "prediction": acc}
        return result
    
    def dummy_detect(self, text, prompt, **kwargs):
        result = {
                  "confidences": 1,
                  "prediction": False}

        return result






