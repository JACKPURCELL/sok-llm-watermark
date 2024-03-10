import argparse
import json
import nltk
import time
import os
import tqdm

import torch
from transformers import SeamlessM4Tv2Model, AutoProcessor

nltk.download("punkt")

def generate_translation(
    data,
    model_name="facebook/seamless-m4t-v2-large",
    start_idx=None,
    end_idx=None,
    args=None,
):
  

    time1 = time.time()
    processor = AutoProcessor.from_pretrained(model_name)
    model = SeamlessM4Tv2Model.from_pretrained(model_name).cuda()
    print("Model loaded in ", time.time() - time1)
    model.eval()

    data = (
        data.select(range(0, len(data)))
        if start_idx is None or end_idx is None
        else data.select(range(start_idx, end_idx))
    )

    w_wm_output_attacked = []
    w_wm_output_frenchs = []
    for idx, dd in tqdm.tqdm(enumerate(data), total=len(data)):


        if "w_wm_output_attacked" not in dd:
            if args.no_wm_attack:
                if isinstance(dd["no_wm_output"], str):
                    input_gen = dd["no_wm_output"].strip()
                else:
                    input_gen = dd["no_wm_output"][0].strip()
            else:
                if isinstance(dd["w_wm_output"], str):
                    input_gen = dd["w_wm_output"].strip()
                else:
                    input_gen = dd["w_wm_output"][0].strip()

           

            
            translated_input_gen =  processor(text=input_gen, src_lang="eng", return_tensors="pt")
            translated_input_gen = {k: v.cuda() for k, v in translated_input_gen.items()}
            
            # Translate to French
            translated_output_tokens = model.generate(
                **translated_input_gen,
                tgt_lang="fra",
                generate_speech=False)
            
            translated_text_french = processor.decode(translated_output_tokens[0].tolist()[0], skip_special_tokens=True)

            tokenized_translated_text_french =  processor(text=translated_text_french, src_lang="fra", return_tensors="pt")
            tokenized_translated_text_french = {k: v.cuda() for k, v in tokenized_translated_text_french.items()}
            
            # Translate back to English
            translated_output_tokens_en = model.generate(
                **tokenized_translated_text_french,
                tgt_lang="eng",
                generate_speech=False)
            translated_text_english = processor.decode(translated_output_tokens_en[0].tolist()[0], skip_special_tokens=True)

            # Append paraphrased text and original text
            w_wm_output_attacked.append(translated_text_english.strip())
            w_wm_output_frenchs.append(translated_text_french)


    data = data.add_column("w_wm_output_attacked", w_wm_output_attacked)
    data = data.add_column("w_wm_output_french", w_wm_output_frenchs)

    return data
