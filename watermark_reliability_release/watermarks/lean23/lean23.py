import json
import torch
import tqdm
from .watermark_processor import RepetitionPenaltyLogitsProcessor
from transformers import LogitsProcessor, LogitsProcessorList, MinLengthLogitsProcessor
from .watermark_processors.message_model_processor import WmProcessorMessageModel
from .watermark_processors.message_models.lm_message_model import LMMessageModel
from transformers import AutoTokenizer, AutoModelForCausalLM


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

        self.watermark_processor.start_length = 50 ### Only for c4
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
    def __init__(self, watermark_processor, generated_length, message_code_len, encode_ratio, message, tokenizer):
        self.watermark_processor = watermark_processor
        self.generated_length = generated_length
        self.message_code_len = message_code_len
        self.encode_ratio = encode_ratio
        self.message = message
        self.tokenizer = tokenizer

    def detect(self, text, prompt, **kwargs):
        if len(self.tokenizer.tokenize(text))<12:
            print("error length@")
            return {
                  "z_score": 0.0,
                  "prediction": False}
        decoded_message, other_information = self.watermark_processor.logit_processor[2].decode(text, disable_tqdm=True)
        confidences = other_information[1]
        available_message_num = self.generated_length // (
            int(self.message_code_len * self.encode_ratio))
        acc = decoded_message[:available_message_num] == self.message[:available_message_num]
        result = {
                  "z_score": confidences[0][2],
                  "prediction": acc}
        return result
    
    def dummy_detect(self, **kwargs):
        result = {
                  "z_score": 0.0,
                  "prediction": False}

        return result

if __name__ == '__main__':
    temperature = 1.0
    model_name='facebook/opt-1.3b'
    sample_num=100
    sample_seed=42
    seed=42
    num_beams=4
    delta=1.5
    repeat_penalty=1.5
    message=[52,12,564,65,67,233]
    prompt_length=300
    generated_length=200
    message_code_len=20
    encode_ratio=10.0
    device='cuda:0'
    root_path='/home/jkl6486/codable-watermarking-for-llm/my_watermark_result'
    wm_strategy='lm_new_7_10'
    lm_prefix_len=10
    lm_top_k=-1
    lm_model_name='gpt2'
    message_model_strategy='vanilla'
    random_permutation_num=100
    max_confidence_lbd=0.5
    top_k=1000
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    watermark_processor = lean23_BalanceMarkingWatermarkLogitsProcessor(tokenizer=tokenizer,
                                                                        lm_tokenizer=tokenizer,
                                                                        lm_model=model,
                                                                        delta=delta,
                                                                        lm_prefix_len=lm_prefix_len,
                                                                        lm_top_k=lm_top_k,
                                                                        message_code_len=message_code_len,
                                                                        random_permutation_num=random_permutation_num,
                                                                        encode_ratio=encode_ratio,
                                                                        max_confidence_lbd=max_confidence_lbd,
                                                                        message_model_strategy=message_model_strategy,
                                                                        message=message,
                                                                        top_k=top_k,
                                                                        repeat_penalty=repeat_penalty
                                                                        )
    watermark_detector = lean23_WatermarkDetector(watermark_processor=watermark_processor,
                                                  generated_length=generated_length,
                                                  message_code_len=message_code_len,
                                                  encode_ratio=encode_ratio,
                                                  tokenizer=tokenizer,
                                                  prompt_length=prompt_length,
                                                  message=message,
                                                  min_prefix_len=10)
    filename = "/home/jkl6486/codable-watermarking-for-llm/gen_table.jsonl"
    with open(filename, "r", encoding="utf-8") as f:
        c4_sliced_and_filted = [json.loads(line) for line in f.read().strip().split("\n")]
        decoded_message_list = []
        other_information_list = []
        for text in c4_sliced_and_filted:
            tokenized_input = tokenizer(text['truncated_input'], return_tensors='pt').to(model.device)
            #tokenized_input = truncate(tokenized_input, max_length=args.prompt_length)

            watermark_processor.logit_processor[2].start_length = tokenized_input['input_ids'].shape[-1]
            output_tokens = model.generate(**tokenized_input,
                                           temperature=temperature,
                                           max_new_tokens=generated_length,
                                           num_beams=num_beams,
                                           logits_processor=[watermark_processor])

            output_text = \
                tokenizer.batch_decode(
                    output_tokens[:, tokenized_input["input_ids"].shape[-1]:],
                    skip_special_tokens=True)[0]

            prefix_and_output_text = tokenizer.batch_decode(output_tokens,
                                                            skip_special_tokens=True)[0]

            decoded_message, other_information = watermark_processor.logit_processor[2].decode(output_text, disable_tqdm=True)
            decoded_message_list.append(decoded_message)
            other_information_list.append(other_information)
            print()
    
    print()





