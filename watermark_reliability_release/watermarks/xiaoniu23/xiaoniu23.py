import torch
from transformers import GenerationMixin, GenerationConfig, pipeline
import types
from unbiased_watermark import patch_model, RobustLLR_Score_Batch_v2

class xiaoniu23_detector():

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def r_llr_score(self, texts, dist_qs, watermark_type, key, **kwargs):

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
        n = 10
        dist_qs = [float(i) / n for i in range(n + 1)]

        labels, _, scores = self.r_llr_score([text], dist_qs=dist_qs, **kwargs)
        import numpy as np

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


        return [best_index, best_dist_q]
    
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

def patch_model(model: GenerationMixin):
    original_generate = model.generate
    original__get_logits_warper = model._get_logits_warper

    context = {}

    def generate(self, *args, logits_warper=None, **kargs):
        logits_warper = (
            logits_warper if logits_warper is not None else LogitsProcessorList()
        )
        context["logits_warper"] = logits_warper
        return original_generate(*args, **kargs)

    def _get_logits_warper(self, *args, **kargs):
        warpers = original__get_logits_warper(*args, **kargs)
        if "logits_warper" in context:
            warpers = self._merge_criteria_processor_list(
                warpers, context["logits_warper"]
            )
        return warpers

    def _clear_patch_context(self):
        context.clear()

    model.generate = types.MethodType(generate, model)
    model._get_logits_warper = types.MethodType(_get_logits_warper, model)
    model._clear_patch_context = types.MethodType(_clear_patch_context, model)

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

class xiaoniu23_generate_with_watermark():
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
        return ids




