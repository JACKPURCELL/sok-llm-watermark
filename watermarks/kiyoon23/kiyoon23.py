import argparse
import os.path
from functools import reduce
from itertools import product
import math
import re
import spacy
import string

import torch

from config import WatermarkArgs, GenericArgs, stop
from models.watermark import InfillModel
from utils.logging import getLogger
from utils.metric import Metric
from torch import cuda
from utils.dataset_utils import preprocess2sentence

from main import str2bool


#@misc{yoo2023robust,
#      title={Robust Multi-bit Natural Language Watermarking through Invariant Features},
#      author={KiYoon Yoo and Wonhyuk Ahn and Jiho Jang and Nojun Kwak},
#      year={2023},
#      eprint={2305.01904},
#      archivePrefix={arXiv},
#      primaryClass={cs.CL}
#}

class kiyoon23():
    """
        Class for embedding watermark in raw text.

        Args:
            tokenizer: The tokenizer object for the main language model (in which we want to inject watermark).
            lm_model: The proxy model.
            lm_tokenizer: The tokenizer of the proxy model.

            strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
            vocab_size: The size of the vocabulary.
            watermark_key: The random seed for the green-listing.
    """

    def __init__(self, debug_mode, dtype, embed, exp_name_generic, exp_name_infill, extract, extract_corrupted, metric_only,
                 num_sample, spacy_model, custom_keywords, do_watermark, eval_only, exclude_cc,
                 keyword_mask, keyword_ratio, mask_order_by, mask_select_method, model_ckpt, model_name,
                 num_epochs, topk, train_infill):
        self.generic_args = argparse.ArgumentParser(description="Generic argument parser")
        self.generic_args.add_argument("--exp_name", type=str, default=exp_name_generic)
        self.generic_args.add_argument("--embed", type=str2bool, default=embed)
        self.generic_args.add_argument("--extract", type=str2bool, default=extract)
        self.generic_args.add_argument("--extract_corrupted", type=str2bool, default=extract_corrupted)
        self.generic_args.add_argument("--dtype", type=str, default=dtype)
        self.generic_args.add_argument("--num_sample", type=int, default=num_sample)
        self.generic_args.add_argument("--spacy_model", type=str, default=spacy_model)
        self.generic_args.add_argument("--debug_mode", type=str2bool, default=debug_mode)
        self.generic_args.add_argument("--metric_only", type=str2bool, default=metric_only)

        self.infill_args = argparse.ArgumentParser(description="For the watermarking module")
        self.infill_args.add_argument("--debug_mode", type=str2bool, default=debug_mode)
        self.infill_args.add_argument("--eval_only", type=str2bool, default=eval_only)
        self.infill_args.add_argument("--do_watermark", type=str2bool, default=do_watermark)
        self.infill_args.add_argument("--train_infill", type=str2bool, default=train_infill)
        self.infill_args.add_argument("--model_ckpt", type=str, nargs="?", const=model_ckpt)
        self.infill_args.add_argument("--model_name", type=str, default=model_name)
        self.infill_args.add_argument("--exp_name", type=str, default=exp_name_infill)
        self.infill_args.add_argument("--num_epochs", type=int, default=num_epochs)
        self.infill_args.add_argument("--dtype", type=str, default=dtype)
        self.infill_args.add_argument("--spacy_model", type=str, default=spacy_model)
        self.infill_args.add_argument("--keyword_ratio", type=float, default=keyword_ratio)
        self.infill_args.add_argument("--topk", type=int, default=topk)
        self.infill_args.add_argument("--mask_select_method", type=str, default=mask_select_method, choices=['keyword_disconnected', "keyword_connected", "grammar"])
        self.infill_args.add_argument("--mask_order_by", type=str, default=mask_order_by, choices=['dep', 'pos'])
        self.infill_args.add_argument("--keyword_mask", type=str, default=keyword_mask, choices=['adjacent', 'child', 'child_dep', "na"])
        self.infill_args.add_argument("--custom_keywords", type=str, default=custom_keywords)
        self.infill_args.add_argument("--exclude_cc", type=str2bool, default=exclude_cc)

        self.infill_args, _ = self.infill_args.parse_known_args()
        self.generic_args, _ = self.generic_args.parse_known_args()

    def generate_with_watermark(self, raw_text):

        cover_texts = preprocess2sentence([raw_text], corpus_name="custom",
                                          start_sample_idx=0, cutoff_q=(0.0, 1.0), use_cache=False)
        # you can add your own entity / keyword that should NOT be masked.
        # This list will need to be saved when extracting
        self.infill_args.custom_keywords = ["watermarking", "watermark"]

        DEBUG_MODE = self.generic_args.debug_mode
        dtype = "custom"
        dirname = f"./results/ours/{dtype}/{self.generic_args.exp_name}"
        start_sample_idx = 0
        num_sample = self.generic_args.num_sample

        spacy_tokenizer = spacy.load(self.generic_args.spacy_model)
        if "trf" in self.generic_args.spacy_model:
            spacy.require_gpu()
        model = InfillModel(self.infill_args, dirname=dirname)

        bit_count = 0
        word_count = 0
        upper_bound = 0
        candidate_kwd_cnt = 0
        kwd_match_cnt = 0
        mask_match_cnt = 0
        sample_cnt = 0
        one_cnt = 0
        zero_cnt = 0

        # select device
        if torch.has_mps:
            device = "mps"
        elif cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        metric = Metric(device, **vars(self.generic_args))

        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        logger = getLogger("DEMO",
                           dir_=dirname,
                           debug_mode=DEBUG_MODE)

        result_dir = os.path.join(dirname, "watermarked.txt")
        if not os.path.exists(result_dir):
            os.makedirs(os.path.dirname(result_dir), exist_ok=True)

        for c_idx, sentences in enumerate(cover_texts):
            corpus_level_watermarks = []
            for s_idx, sen in enumerate(sentences):
                sen = spacy_tokenizer(sen.text.strip())
                all_keywords, entity_keywords = model.keyword_module.extract_keyword([sen])
                keyword = all_keywords[0]
                ent_keyword = entity_keywords[0]

                agg_cwi, agg_probs, tokenized_pt, (mask_idx_pt, mask_idx, mask_word) = model.run_iter(sen, keyword,
                                                                                                      ent_keyword,
                                                                                                      train_flag=False,
                                                                                                      embed_flag=True)
                # check if keyword & mask_indices matches
                valid_watermarks = []
                candidate_kwd_cnt = 0
                tokenized_text = [token.text_with_ws for token in sen]

                if len(agg_cwi) > 0:
                    for cwi in product(*agg_cwi):
                        wm_text = tokenized_text.copy()
                        for m_idx, c_id in zip(mask_idx, cwi):
                            wm_text[m_idx] = re.sub(r"\S+", model.tokenizer.decode(c_id), wm_text[m_idx])

                        wm_tokenized = spacy_tokenizer("".join(wm_text).strip())

                        # extract keyword of watermark
                        wm_keywords, wm_ent_keywords = model.keyword_module.extract_keyword([wm_tokenized])
                        wm_kwd = wm_keywords[0]
                        wm_ent_kwd = wm_ent_keywords[0]
                        wm_mask_idx, wm_mask = model.mask_selector.return_mask(wm_tokenized, wm_kwd, wm_ent_kwd)

                        kwd_match_flag = set([x.text for x in wm_kwd]) == set([x.text for x in keyword])
                        if kwd_match_flag:
                            kwd_match_cnt += 1

                        # checking whether the watermark can be embedded without the assumption of corruption
                        mask_match_flag = len(wm_mask) and set(wm_mask_idx) == set(mask_idx)
                        if mask_match_flag:
                            text2print = [t.text_with_ws for t in wm_tokenized]
                            for m_idx in mask_idx:
                                text2print[m_idx] = f"\033[92m{text2print[m_idx]}\033[00m"
                            valid_watermarks.append(text2print)
                            mask_match_cnt += 1

                        sample_cnt += 1
                        candidate_kwd_cnt += 1

                punct_removed = sen.text.translate(str.maketrans(dict.fromkeys(string.punctuation)))
                word_count += len([i for i in punct_removed.split(" ") if i not in stop])
                print(f"***Sentence {s_idx}***")
                if len(valid_watermarks) > 1:
                    bit_count += math.log2(len(valid_watermarks))
                    for vw in valid_watermarks:
                        print("".join(vw))

                if len(valid_watermarks) == 0:
                    valid_watermarks = [sen.text]

                corpus_level_watermarks.append(valid_watermarks)

                if candidate_kwd_cnt > 0:
                    upper_bound += math.log2(candidate_kwd_cnt)

            if word_count:
                logger.info(f"Bpw : {bit_count / word_count:.3f}")

            num_options = reduce(lambda x, y: x * y, [len(vw) for vw in corpus_level_watermarks])
            available_bit = math.floor(math.log2(num_options))
            print(num_options)
            print(f"Input message to embed (max bit:{available_bit}):")
            message = input().replace(" ", "")
            # left pad to available bit if given message is short
            if available_bit > 8:
                print(f"Available bit is large: {available_bit} > 8.. "
                      f"We recommend using shorter text segments as it may take a while")
            message = "0" * (available_bit - len(message)) + message
            if len(message) > available_bit:
                print(f"Given message longer than capacity. Truncating...: {len(message)}>{available_bit}")
                message = message[:available_bit]
            message_decimal = int(message, 2)
            # breakpoint()
            cnt = 0
            available_candidates = product(*corpus_level_watermarks)
            watermarked_text = next(available_candidates)
            while cnt < message_decimal:
                cnt += 1
                watermarked_text = next(available_candidates)

            print("---- Watermarked text ----")
            for wt in watermarked_text:
                print("".join(wt))



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
            return {"decoded_message": "\n\n212 shares",
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
        result = {"decoded_message": decoded_message,
                  "confidences": confidences,
                  "prediction": acc}
        return result






