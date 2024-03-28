# Implementaion of watermark method presented in "Watermarking GPT Outputs" from Scott Aaronson (UT Austin and OpenAI) and Henarik Kirchner (OpenAI)
# Coded by Alps lab in Stony Brook University
from __future__ import annotations
import torch
from transformers import LogitsProcessor, pipeline, OPTForCausalLM, AutoTokenizer, LogitsProcessorList
import hashlib
import numpy as np
from math import log



class scott22_WatermarkLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        key: int, ### The key for generating the watermark
        window_size: int, ### The size of the window for watermarking
    ):
        self.key = key
        self.window_size = window_size
        self.rng = None


    def _seed_rng(self, input_ids: torch.LongTensor) -> None:
        # can optionally override the seeding scheme,
        # but uses the instance attr by default
        self.rng = torch.Generator(device=input_ids.device)

        if seeding_scheme is None:
            seeding_scheme = self.seeding_scheme

        if seeding_scheme == "simple_1":
            assert input_ids.shape[-1] >= 1, f"seeding_scheme={seeding_scheme} requires at least a 1 token prefix sequence to seed rng"
            prev_token = input_ids[-1].item()
            self.rng.manual_seed(self.hash_key * prev_token)
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {seeding_scheme}")
        return

    def _get_r_value(self, input_ids: torch.LongTensor) -> float:
        token_strings = input_ids.tolist()
        r_list = []
        for token_string in token_strings:
            token_string = ",".join(map(str, token_string))+","+str(self.key) ### We append the secret key at the end
            hash_obj = hashlib.sha256(token_string.encode('utf-8'))
            hash_digest = hash_obj.digest()
            hash_int = int.from_bytes(hash_digest, 'big')
            max_int = 2**256 - 1
            r = hash_int / max_int
            r_list.append(r)
        return r_list
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # self._seed_rng(input_ids)
        # input_ids_append = torch.cat((input_ids_append, idx), dim=1)
        
        # prf_key = self.key * input_ids[-self.window_size :].sum().item()
        # self.rng.manual_seed(prf_key % (2**64 - 1))  # safeguard against overflow from long

        # # enable for long, interesting streams of pseudorandom numbers: print(prf_key)
        # self.rng.manual_seed(prf_key % (2**64 - 1))  # safeguard against overflow from long
        # greenlist_size = int(self.vocab_size * self.gamma)
        # vocab_permutation = torch.randperm(self.vocab_size, device=input_ids.device, generator=self.rng)
        self.rng = torch.Generator(device=input_ids.device)
    

        scores = torch.nn.functional.softmax(scores, dim=-1)
        if self.window_size > input_ids.shape[1]:
            return scores
        rk = self.key * input_ids[:,-self.window_size:].sum().item()
        self.rng.manual_seed(rk)
        r_i = torch.rand(scores.shape[1], device=input_ids.device, generator=self.rng)
        # input_ids_append = input_ids.repeat(scores.shape[1], 1)
        # idx = torch.LongTensor(range(scores.shape[1])).to(input_ids.device)
        # idx = idx.view(-1, 1)
        # input_ids_append = torch.cat((input_ids_append, idx), dim=1)
        # input_ids_append = input_ids_append[:,-self.window_size :].sum(axis=1)
        # prf_key = self.key * input_ids_append
        # r_list = []
        # for i in prf_key:
        #     r_i = torch.rand(1, device=input_ids.device, generator=self.rng.manual_seed(i.item()))
        #     r_list.append(r_i.item())
        #r_list = torch.Tensor(r_list).to(input_ids.device)

        scores = torch.pow(r_i, 1/scores[0])
        # print(r_i[torch.argmax(scores)])
        # print(torch.argmax(scores))
        # print(rk)
        # print(input_ids[:,-self.window_size:])


        # r_list = self._get_r_value(input_ids_append[:, -self.window_size:])
        # r_list = torch.Tensor(r_list).to(input_ids.device)
        # scores = torch.pow(r_list, 1/scores[0])
        scores = torch.unsqueeze(scores, 0)
        return scores
    

class scott22_WatermarkDetector():
    def __init__(self, key, window_size, tokenizer, vocab_size, device, threshold=0):
        self.key = key
        self.window_size = window_size
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.vocab_size = vocab_size
        self.device = device




    def _get_r_value(self, input_ids: torch.LongTensor) -> float:
        token_string = input_ids.tolist()
        r_list = []
        token_string = ",".join(map(str, token_string))+","+str(self.key) ### We append the secret key at the end
        hash_obj = hashlib.sha256(token_string.encode('utf-8'))
        hash_digest = hash_obj.digest()
        hash_int = int.from_bytes(hash_digest, 'big')
        max_int = 2**256 - 1
        r = hash_int / max_int
        r_list.append(r)
        return r_list
        
    def detect(self, text, prompt, **kwargs):
        self.rng = torch.Generator(device=self.device)
        r_list = []
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)
        for t in range(self.window_size, input_ids.shape[1]-1):
            if t < self.window_size:
                continue
            rk = input_ids[0][t-self.window_size+1:t+1].sum().item() * self.key
            self.rng.manual_seed(rk)
            r_i = torch.rand(self.vocab_size, device=input_ids.device, generator=self.rng)
            r_list.append(r_i[input_ids[0][t+1]])
            #print(r_i[input_ids[0][t+1]])
            # r_list += self._get_r_value(input_ids[0][t-self.window_size:t])
            # prf = input_ids[0][t-self.window_size:t].sum() * self.key
            # r_list.append(torch.rand(1, device="cuda", generator=self.rng.manual_seed(prf.item())))
        z_score = (sum([log(1/(1-r)) for r in r_list])/(len(r_list)-self.window_size))-1
        detection = {"z_score": z_score,
                     "prediction": z_score > self.threshold}
        return detection
    
    def dummy_detect(self, **kwargs):
        return{"z_score": float("nan"),
               "prediction": False}

if __name__ == "__main__":
    torch.set_default_device("cuda")
    prompt = "Hello, who are you?"
    model_name = "facebook/opt-1.3b"
    model = OPTForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    lp = scott22_WatermarkLogitsProcessor(key=123, window_size=4)
    lp_list = LogitsProcessorList([lp])
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_length=300,
            num_return_sequences=1,
            logits_processor=lp_list)
        outputs2 = model.generate(
            input_ids=input_ids,
            max_length=300,
            num_return_sequences=1)

    d = scott22_WatermarkDetector(key=123, window_size=4, tokenizer=tokenizer, device = "cuda", vocab_size=50272)
    o = tokenizer.decode(outputs[0][input_ids[0].shape[0]:])
    o2 = tokenizer.decode(outputs2[0][input_ids[0].shape[0]:])
    print("----------------------------")
    a = d.detect(o, prompt)
    b = d.detect(o2,prompt)
    print()






