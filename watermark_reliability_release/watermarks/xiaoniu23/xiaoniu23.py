import random
import torch
from torch import tensor
from transformers import GenerationConfig, is_torch_available
from .unbiased_watermark import patch_model, RobustLLR_Score_Batch_v2, WatermarkLogitsProcessor, Delta_Reweight, Gamma_Reweight, PrevN_ContextCodeExtractor
from math import log
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LogitsProcessorList, pipeline


#@inproceedings{
#anonymous2023unbiased,
#title={Unbiased Watermark for Large Language Models},
#author={Anonymous},
#booktitle={Submitted to The Twelfth International Conference on Learning Representations},
#year={2023},
#url={https://openreview.net/forum?id=uWVC5FVidc},
#note={under review}
#}

cache = {
    "model_str": None,
    "generator": None,
    "tokenizer": None,
}

def get_prompt_length(tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    return inputs["input_ids"].shape[1]

def get_threshold(n, alpha):
    return -1*log(alpha)+log(n)

def load_model(model_str, num_beams, temps, top_p):
    if model_str == cache["model_str"]:
        return cache
    else:
        if temps is None:
            generator = pipeline(
                "text-generation",
                model=model_str,
                do_sample=True,
                num_beams=num_beams,
                device_map='auto',
            )
        else:
            generator = pipeline(
                "text-generation",
                model=model_str,
                do_sample=True,
                num_beams=num_beams,
                device_map='auto',
                temperature=temps,
                top_p=top_p,
            )

        cache["model_str"] = model_str
        cache["generator"] = generator
        cache["tokenizer"] = generator.tokenizer
        patch_model(cache["generator"].model)
        cache["tokenizer"].add_special_tokens({'pad_token': '[PAD]'})
        return cache
    
def get_wp(watermark_type, key):

    if watermark_type == "delta":
        rw = Delta_Reweight()
    elif watermark_type == "gamma":
        rw = Gamma_Reweight()
    else:
        raise ValueError(f"Unknown watermark type: {watermark_type}")
    wp = WatermarkLogitsProcessor(key, rw, PrevN_ContextCodeExtractor(5))
    return wp

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




class xiaoniu23_detector():
    def __init__(self, model_name, n, alpha, private_key, watermark_type, num_beams, tokenizer, temperature, top_p):
        self.model_name = model_name
        self.n = n
        self.alpha = alpha
        self.threshold = get_threshold(n, alpha)
        self.private_key = private_key.encode("utf-8")
        self.watermark_type = watermark_type
        self.tokenizer = tokenizer
        self.num_beams = num_beams
        self.temperature = temperature
        self.top_p = top_p
    


    def r_llr_score(self, model_str, texts, dist_qs, watermark_type, key):
        score = RobustLLR_Score_Batch_v2.from_grid([0.0], dist_qs)
        wp = get_wp(watermark_type, key)
        wp.ignore_history = True
        cache = load_model(model_str, num_beams=self.num_beams, temps=self.temperature, top_p=self.top_p)
        inputs = cache["tokenizer"](texts, return_tensors="pt", padding=True)


        model = cache["generator"].model
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
#TODO fix align with parameters
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
        model_str = self.model_name
        prompt_len = get_prompt_length(self.tokenizer, prompt)
        compute_range = (prompt_len, None)
        n = self.n
        dist_qs = [float(i) / n for i in range(n + 1)]

        labels, _, scores = self.r_llr_score(model_str, [prompt+text], dist_qs=dist_qs, watermark_type=self.watermark_type, key=self.private_key)


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
        #print("best_dist_q:", best_dist_q)
        #print("best_sum_score:", sum_scores[best_index])
        best_sum_score = sum_scores[best_index]
        if best_sum_score > self.threshold:
            watermarked = True
        else:
            watermarked = False
        return {"best_sum_score": best_sum_score, 
                "best_dist_q": best_dist_q, 
                "prediction": watermarked}
    def dummy_detect(self, **kwargs):
        result = {
                  "best_sum_score": 0.0,
                  "best_dist_q": 0.0,
                  "prediction": False}
        return result
def generate_with_watermark(model_str, input_ids, wp, **kwargs):
    if "num_beams" not in kwargs:
        kwargs["num_beams"] = 1
    if "temperature" not in kwargs:
        kwargs["temperature"] = None
    cache = load_model(model_str, kwargs["num_beams"], kwargs["temperature"], kwargs["top_p"])
    generator = cache["generator"]
    prompt = [cache["tokenizer"].decode(i) for i in input_ids]
    kwargs = {"max_new_tokens": kwargs["max_new_tokens"]}
    outputs = generator(prompt, logits_warper=wp, **kwargs)
    #generator.model._clear_patch_context()
    outputs = [r[0]["generated_text"] for r in outputs]
    return tensor(cache["tokenizer"](outputs, padding=True)["input_ids"]).cuda()
    
if __name__ == "__main__":
    prompt = "What is a watermark? What's the purpose of it?"
    model_name = "facebook/opt-6.7b"
    key = b'private key'
    raw_key = 'private key'
    '''model = AutoModelForCausalLM.from_pretrained('facebook/opt-6.7b')
    #patch_model(model)
    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-6.7b')
    inputs = tokenizer([prompt], return_tensors="pt")
    reweight_delta = Delta_Reweight()
    reweight_gamma = Gamma_Reweight()

    watermark_processor = WatermarkLogitsProcessor(
        key,
        reweight_delta,
        PrevN_ContextCodeExtractor(5))
    
    output2 = generate_with_watermark(
        model_name, [prompt], wp=[watermark_processor], max_length=100
    )[0]

    patch_model(model)
    output_ids = model.generate(
        inputs["input_ids"],
        max_length=128,
        do_sample=True,
        num_beams=1,
        top_k=0,
        temperature=1.0,
        logits_warper=[watermark_processor],
    )'''
    
    ################################################################
    output2 = "</s>What is a watermark? What's the purpose of it?\nIt is supposed to be seen by you only and not everyone else, but if they use a tool such as photoshop (which I can’t), they can put it on. It basically makes it easier to find their work\nOh okay. I thought it was something important I missed and we were supposed to know about it. Now I see it."
    detector = xiaoniu23_detector('facebook/opt-6.7b', 5, 0.05, raw_key, "delta")
    result2 = detector.detect(output2, prompt)    
    
    
    print()







