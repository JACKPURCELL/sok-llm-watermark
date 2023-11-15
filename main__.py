import os, argparse
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from mersenne import mersenne_rng

import numpy as np
import torch
from transformers import AutoTokenizer
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test for a watermark in a text document')
    parser.add_argument('document',type=str, help='a file containing the document to test')
    parser.add_argument('--tokenizer',default='facebook/opt-1.3b',type=str,
            help='a HuggingFace model id of the tokenizer used by the watermarked model')
    parser.add_argument('--dataset',default='??',type=str,
            help='a HuggingFace model id of the tokenizer used by the watermarked model')
    parser.add_argument('--n',default=256,type=int,
            help='the length of the watermark sequence')
    parser.add_argument('--key',default=42,type=int,
            help='the seed for the watermark sequence')




    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count > 0:
        device = torch.device('cuda')
        print("DEVICE FOUND: %s" % device)
        parallel = True if torch.cuda.device_count > 1 else False
    else:
        device = torch.device('cpu')


    # Set args.seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)

    tokens = tokenizer.encode(args.prompt, return_tensors='pt', truncation=True, max_length=2048)

    watermarked_tokens = generate_shift(model,tokens,len(tokenizer),args.n,args.m,args.key)[0]
    watermarked_text = tokenizer.decode(watermarked_tokens, skip_special_tokens=True)



    generate_without_watermark = partial(
        model.generate,
        **gen_kwargs
    )
    generate_with_watermark = partial(
        model.generate,
        logits_processor=LogitsProcessorList([watermark_processor]), 
        **gen_kwargs
    )
    if args.prompt_max_length:
        pass
    elif hasattr(model.config,"max_position_embedding"):
        args.prompt_max_length = model.config.max_position_embeddings-args.max_new_tokens
    else:
        args.prompt_max_length = 2048-args.max_new_tokens

    tokd_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=args.prompt_max_length).to(device)
    # tokens = tokenizer.encode(args.prompt, return_tensors='pt', truncation=True, max_length=2048)
    
    truncation_warning = True if tokd_input["input_ids"].shape[-1] == args.prompt_max_length else False
    redecoded_input = tokenizer.batch_decode(tokd_input["input_ids"], skip_special_tokens=True)[0]

    torch.manual_seed(args.generation_seed)
    output_without_watermark = generate_without_watermark(**tokd_input)

    # optional to seed before second generation, but will not be the same again generally, unless delta==0.0, no-op watermark
    if args.seed_separately: 
        torch.manual_seed(args.generation_seed)
    output_with_watermark = generate_with_watermark(**tokd_input)
    # watermarked_tokens = generate_shift(model,tokens,len(tokenizer),args.n,args.m,args.key)[0]
    
    if args.is_decoder_only_model:
        # need to isolate the newly generated tokens
        output_without_watermark = output_without_watermark[:,tokd_input["input_ids"].shape[-1]:]
        output_with_watermark = output_with_watermark[:,tokd_input["input_ids"].shape[-1]:]

    decoded_output_without_watermark = tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0]
    decoded_output_with_watermark = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0]
    # watermarked_text = tokenizer.decode(watermarked_tokens, skip_special_tokens=True)
    

    return (redecoded_input,
            int(truncation_warning),
            decoded_output_without_watermark, 
            decoded_output_with_watermark,
            args) 
            # decoded_output_with_watermark)
            