import argparse
import os.path
from functools import reduce
from itertools import product
import math
import re
import spacy
import string

import torch

from models.watermark import InfillModel
from utils.logging import getLogger
from utils.metric import Metric
from torch import cuda
from utils.dataset_utils import preprocess2sentence


from nltk.corpus import stopwords




#@misc{yoo2023robust,
#      title={Robust Multi-bit Natural Language Watermarking through Invariant Features},
#      author={KiYoon Yoo and Wonhyuk Ahn and Jiho Jang and Nojun Kwak},
#      year={2023},
#      eprint={2305.01904},
#      archivePrefix={arXiv},
#      primaryClass={cs.CL}
#}

stop = set(stopwords.words('english'))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class kiyoon23():
    """
        Class for embedding watermark in raw text.

        Args:
            SPACYM: type of spacy model used for dependency parsing
            KR: keyword ratio that determines the number of keywords and masks (see Table 11 for configuration)
            TOPK: topk infill words used to infill selected masks (see Table 11 for configuration)
            MASK_S: mask selection method, choose from {keyword_connected, grammar}
            MASK_ORDR_BY: ordering of masks by {dep, pos}. This is only relevant when using dependency component
            EXCLUDE_CC: exlucde the cc dependency as detailed in Section 5.2
            K_MASK: how mask is selected when using keyword component; only relvant when using keyword component, choose from {adjacent, child}

    """

    def __init__(self, dtype, embed, exp_name_generic, exp_name_infill, extract, num_sample, spacy_model, exclude_cc,
                 custom_keywords, keyword_mask, keyword_ratio, mask_order_by, mask_select_method, num_epochs, topk, message):
        self.generic_args = argparse.ArgumentParser(description="Generic argument parser")
        self.generic_args.add_argument("--exp_name", type=str, default=exp_name_generic)
        self.generic_args.add_argument("--embed", type=str2bool, default=embed)
        self.generic_args.add_argument("--extract", type=str2bool, default=extract)
        self.generic_args.add_argument("--extract_corrupted", type=str2bool, default=False)
        self.generic_args.add_argument("--dtype", type=str, default=dtype)
        self.generic_args.add_argument("--num_sample", type=int, default=num_sample)
        self.generic_args.add_argument("--spacy_model", type=str, default=spacy_model)
        self.generic_args.add_argument("--debug_mode", type=str2bool, default=False)
        self.generic_args.add_argument("--metric_only", type=str2bool, default=False)

        self.infill_args = argparse.ArgumentParser(description="For the watermarking module")
        self.infill_args.add_argument("--debug_mode", type=str2bool, default=False)
        self.infill_args.add_argument("--eval_only", type=str2bool, default=False)
        self.infill_args.add_argument("--do_watermark", type=str2bool, default=True)
        self.infill_args.add_argument("--train_infill", type=str2bool, default=False)
        self.infill_args.add_argument("--model_ckpt", type=str, nargs="?", const=None)
        self.infill_args.add_argument("--model_name", type=str, default="bert-large-cased")
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
        self.message = message

    def embed_watermark(self, raw_text, message):

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

        result = ""
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
            message = message.replace(" ", "")
            # left pad to available bit if given message is short
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

            for wt in watermarked_text:
                result = result.join(wt).join(" ")

        ### Delete the last blank we addded
        result = result.strip()

        return result

    def extract_message(self, watermarked_text):

        cover_texts = preprocess2sentence([watermarked_text], corpus_name="custom",
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

        result = ""
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
                                text2print[m_idx] = f"\033[92m{text2print[m_idx]}\033[00m" ### Check here
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
            cnt = 0
            while watermarked_text != corpus_level_watermarks:
                cnt += 1
            message = str(bin(cnt))[2:]
            return message





    def generate_with_watermark(self, tokd_input, generate_without_watermark):
        output_without_watermark = generate_without_watermark(tokd_input)
        output_with_watermark = self.embed_watermark(output_without_watermark, self.message)
        return output_with_watermark

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name_generic", type=str, default="tmp")
    parser.add_argument("--embed", type=str2bool, default=False)
    parser.add_argument("--extract", type=str2bool, default=False)
    parser.add_argument("--dtype", type=str, default="imdb")
    parser.add_argument("--num_sample", type=int, default=100)
    parser.add_argument("--exp_name_infill", type=str, default="")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--keyword_ratio", type=float, default=0.05)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--mask_select_method", type=str, default="grammar",
                        choices=['keyword_disconnected', "keyword_connected", "grammar"])
    parser.add_argument("--mask_order_by", type=str, default="dep", choices=['dep', 'pos'])
    parser.add_argument("--keyword_mask", type=str, default="adjacent",
                        choices=['adjacent', 'child', 'child_dep', "na"])
    parser.add_argument("--custom_keywords", type=str, default=['watermarking', 'watermark'])
    parser.add_argument("--message", type=str, default="")
    parser.add_argument("--spacy_model", type=str, default="en_core_web_sm")
    parser.add_argument("--exclude_cc", type=str2bool, default=True)
    args = parser.parse_args()
    watermark_processor = kiyoon23(args.dtype, args.embed, args.exp_name_generic, args.exp_name_infill,
                                              args.extract,
                                              args.num_sample, args.spacy_model, args.exclude_cc, args.custom_keywords,
                                              args.keyword_mask,
                                              args.keyword_ratio, args.mask_order_by, args.mask_select_method,
                                              args.num_epochs,
                                              args.topk, 101)
    raw_text = """
    The White House said it is monitoring the shooting reported at the University of Nevada, Las Vegas (UNLV) “very closely.” The second gentleman is already scheduled to deliver remarks tonight at the Newtown Action Alliance Foundation’s 11th Annual National Vigil for All Victims of Gun Violence, the White House added.
    """
    wm_text = watermark_processor.embed_watermark(raw_text=raw_text, message="101")
    decoded_msg = watermark_processor.extract_message(wm_text)
    print()









