### Comparing Perplexity of different models
### From Hugginface: https://huggingface.co/docs/transformers/perplexity
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from tqdm import tqdm
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,3"
    device = "cuda"
    model_id = "facebook/opt-1.3b"
    # model_id = "meta-llama/Llama-2-7b-chat-hf"
    # facebook/opt-1.3b
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", ignore_verifications=True)
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt").to(device)
    max_length = model.config.max_position_embeddings
    stride = 512
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0
    counter = 0 
    for begin_loc in tqdm(range(0, seq_len, stride)):
        if counter >= 500:  # Stop the loop if counter reaches 1000
            break
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone().to(device)
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
        counter += 1         
    ppl = torch.exp(torch.stack(nlls).mean())
    print(ppl)
    print()