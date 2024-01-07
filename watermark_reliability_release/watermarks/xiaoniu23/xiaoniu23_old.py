import random
import torch
from transformers import GenerationConfig, is_torch_available
from unbiased_watermark import patch_model, RobustLLR_Score_Batch_v2
from math import log
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from unbiased_watermark import patch_model
import completion_demo

#@inproceedings{
#anonymous2023unbiased,
#title={Unbiased Watermark for Large Language Models},
#author={Anonymous},
#booktitle={Submitted to The Twelfth International Conference on Learning Representations},
#year={2023},
#url={https://openreview.net/forum?id=uWVC5FVidc},
#note={under review}
#}

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available

def get_prompt_length(tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    return inputs["input_ids"].shape[1]

def get_wp(watermark_type, key):
    from unbiased_watermark import (
        WatermarkLogitsProcessor,
        Delta_Reweight,
        Gamma_Reweight,
        PrevN_ContextCodeExtractor,
    )

    if watermark_type == "delta":
        rw = Delta_Reweight()
    elif watermark_type == "gamma":
        rw = Gamma_Reweight()
    else:
        raise ValueError(f"Unknown watermark type: {watermark_type}")
    wp = WatermarkLogitsProcessor(key, rw, PrevN_ContextCodeExtractor(5))
    return wp



class xiaoniu23_detector():
    """
    #     Args:
    #         model: The model to be injected. Must gone through patch_model().
    #         tokenizer: The tokenizer used to tokenize the prompt.
    #         n: The number of candidate values for d (the perturbation strength) being considered in your grid search.
    #         alpha: the desired upper bound of the type I error rate.
    #         private_key: The private key used to generate the watermark.
    #         watermark_type: The type of watermark. Can be "delta" or "gamma".
    #     """
    def __init__(self, model, tokenizer, n, alpha, private_key, watermark_type):
        self.model = model
        self.tokenizer = tokenizer
        self.n = n
        self.alpha = alpha
        self.threshold = self.get_threshold(n, alpha)
        self.private_key = private_key.encode("utf-8")
        self.watermark_type = watermark_type

    def set_seed(self, seed):
        set_seed(seed)

    def set_threshold(self, n, alpha):
        self.threshold = self.get_threshold(n, alpha)

    def get_threshold(self, n, alpha):
        return -1*log(alpha)+log(n)

    def r_llr_score(self, texts, dist_qs, watermark_type, key):

        score = RobustLLR_Score_Batch_v2.from_grid([0.0], dist_qs)
        wp = get_wp(watermark_type, key)
        wp.ignore_history = True

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True)

        model = self.model
        input_ids = inputs["input_ids"][..., :-1].to(model.device)
        attention_mask = inputs["attention_mask"][..., :-1].to(model.device)
        labels = inputs["input_ids"][..., 1:].to(model.device)
        labels_mask = inputs["attention_mask"][..., 1:].to(model.device)
        generation_config = GenerationConfig.from_model_config(model.config)
        logits_processor = model._get_logits_processor(
            generation_config,
            input_ids_seq_length=input_ids.shape[-1],
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=None,
            logits_processor=[],
        )
        logits_warper = model._get_logits_warper(generation_config)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits
        old_logits = torch.clone(logits)
        new_logits = torch.clone(logits)
        for i in range(logits.size(1)):
            pre = input_ids[:, : i + 1]
            t = logits[:, i]
            t = logits_processor(pre, t)
            t = logits_warper(pre, t)
            old_logits[:, i] = t
            new_logits[:, i] = wp(pre, t)
        llr, max_llr, min_llr = score.score(old_logits, new_logits)
        query_ids = labels
        unclipped_scores = torch.gather(llr, -1, query_ids.unsqueeze(-1)).squeeze(-1)
        # scores : [batch_size, input_ids_len, query_size]
        scores = torch.clamp(unclipped_scores.unsqueeze(-1), min_llr, max_llr)
        return labels, labels_mask, scores * labels_mask.unsqueeze(-1)

    def detect(self, text, prompt, **kwargs):
        n = self.n
        prompt_len = get_prompt_length(self.tokenizer, prompt)
        compute_range = (prompt_len, None)
        dist_qs = [float(i) / n for i in range(n + 1)]

        labels, _, scores = self.r_llr_score([text], 
                                             dist_qs=dist_qs, 
                                             watermark_type=self.watermark_type, 
                                             key=self.private_key)

        labels = np.array(labels[0].cpu())
        scores = np.array(scores[0].cpu())
        if compute_range[0] is None:
            compute_range = (0, compute_range[1])
        if compute_range[1] is None:
            compute_range = (compute_range[0], len(labels))
        scores[: compute_range[0], :] = 0
        scores[compute_range[1] :, :] = 0
        sum_scores = np.sum(scores, axis=0)
        best_index = np.argmax(sum_scores)
        best_dist_q = dist_qs[best_index]
        best_sum_score = sum_scores[best_index]
        #print("best_dist_q:", best_dist_q)
        #print("best_sum_score:", sum_scores[best_index])
        if best_sum_score > self.threshold:
            watermarked = True
        else:
            watermarked = False

        return [best_index, best_dist_q, best_sum_score, watermarked]
    
        '''tokenizer = self.tokenizer
        result = []
        i = 0
        while i < len(labels):
            for j in range(i + 1, len(labels) + 1):
                token_id = labels[i:j]
                token = tokenizer.decode(token_id, skip_special_tokens=False)
                if merge_till_displayable and not is_displayable(token):
                    continue
                break
            if j < compute_range[0] or i >= compute_range[1]:
                result.append((token_id, token, None))
            else:
                result.append((token_id, token, scores[i:j, best_index].sum()))
            i = j

        if not show_latex:
            print(result)
        else:
            print_latex(result)'''

'''class xiaoniu23_generate_with_watermark():
    def __init__(self, model_str, tokenizer):
        generator = load_model(model_str)
        patch_model(generator.model)
        self.generator_pipeline = generator
        self.tokenizer = tokenizer

    def xiaoniu23_generate_with_watermark(self, prompt, watermark_type, key, **kwargs):
        ### Return a list of tensors, each tensor is the ids of generated text.
        generator = self.generator_pipeline
        if watermark_type is None:
            lws = []
        else:
           lws = [get_wp(watermark_type, key)]
        outputs = generator(prompt, logits_warper=lws, **kwargs)
        generator.model._clear_patch_context()
        ids = [self.tokenizer.encode(r[0]["generated_text"], return_tensor='pt') for r in outputs]
        return ids'''
    
if __name__ == "__main__":
    prompt = "What is a watermark? What's the purpose of it?"
    set_seed(42)
    key = b'private key'
    raw_key = 'private key'
    model = AutoModelForCausalLM.from_pretrained('facebook/opt-6.7b')
    patch_model(model)
    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-6.7b')
    '''reweight_delta = Delta_Reweight()
    reweight_gamma = Gamma_Reweight()

    watermark_processor = WatermarkLogitsProcessor(
        key,
        reweight_gamma,
        PrevN_ContextCodeExtractor(5))
    
    inputs = tokenizer([prompt], return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100, return_dict_in_generate=True, logits_warper=[watermark_processor], num_beams=1)
    '''
    detector = xiaoniu23_detector(model, tokenizer, 5, 0.05, raw_key, "delta")
    output = "What is a watermark? What's the purpose of it?\nIt is supposed to be watermarking the pictures that you took with your phone i think. So, so you can share your pictures and not take credit for them."
    result = detector.detect(output, prompt)
    
    ###############################################
    
    model_name = "facebook/opt-6.7b"
    prompt_len = completion_demo.get_prompt_length(model_name, prompt)
    completion_demo.show_r_llr_score(
        model_name,
        output,
        compute_range=(prompt_len, None),
        show_latex=True,
        watermark_type="delta",
        key=key,
    )

    
    
    
    
    print()







