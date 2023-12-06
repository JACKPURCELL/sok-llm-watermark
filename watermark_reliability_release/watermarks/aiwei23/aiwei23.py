from __future__ import annotations
import torch
from transformers import (GPT2Tokenizer,
                          GPT2LMHeadModel,
                          AutoTokenizer,
                          AutoModelForCausalLM,
                          LlamaTokenizer,
                          LogitsProcessor,
                          LogitsProcessorList)
import json
import random
from tqdm import tqdm
import copy
import os
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from functools import partial
from .util import (get_model,
                  int_to_bin_list,
                  load_data,
                  max_number,
                  get_value,
                  prepare_data,
                  Seq2SeqDataset,
                  TransformerClassifier,
                  train_collate_fn,
                  test_collate_fn)
from math import sqrt


# @misc{liu2023private,
#       title={A Private Watermark for Large Language Models},
#       author={Aiwei Liu and Leyi Pan and Xuming Hu and Shu'ang Li and Lijie Wen and Irwin King and Philip S. Yu},
#       year={2023},
#       eprint={2307.16230},
#       archivePrefix={arXiv},
#       primaryClass={cs.CL}
# }

def prepare_generator(bit_number, sample_number, window_size, layers):
    """
            Function for preparing data and training generator.

            Args:
                bit_number: The number of encoding bits for the model (>log(len(tokenizer))).
                sample_number: The number of generated samples for training the watermark generator.
                window_size: The window size of the method.
                layers: The number of layers of the watermark generator model.
        """
    data_dir = "./data/train_generator_data/train_generator_data.jsonl"
    model_dir = "./model/"
    print("Generating training data for watermark generator.")
    generate_model_key_data(bit_number, sample_number, data_dir, window_size)
    print("Training watermark generator.")
    train_model_generator(data_dir, bit_number, model_dir, window_size, layers)

def generate_model_key_data(bit_number, sample_number, output_file, window_size):
    # We need to generate pairs for all numbers from 0 to 15 (inclusive)
    numbers = list(range(1, max_number(bit_number)))
    # process the pairs and assign balanced labels
    data = []
    combined_set = set()  # Use a set to track unique combined data
    # Iterating over all pairs of numbers
    for _ in tqdm(range(sample_number)):
        # create a list of labels for each num, half 0s and half 1s
        labels = [0, 1]
        random.shuffle(labels)
        combined = []
        # Loop over window size
        for _ in range(window_size - 1):
            # random pick number from numbers and ensure unique
            num = random.choice(numbers)
            bin_num = int_to_bin_list(num, bit_number)
            combined.append(bin_num)

        for label in labels:
            combined1 = copy.deepcopy(combined)
            num = random.choice(numbers)
            bin_num = int_to_bin_list(num, bit_number)
            # import ipdb; ipdb.set_trace()
            combined1.append(bin_num)
            # assign the label
            data.append({"data": combined1, "label": label})
    # save to jsonl
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry))
            f.write('\n')

def train_model_generator(data_dir, bit_number, model_dir, window_size, layers):
    model = get_model(bit_number, window_size, None, layers)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    features, labels = load_data(data_dir)

    # Convert the data into tensor datasets
    train_data = TensorDataset(torch.from_numpy(np.array(features)), torch.from_numpy(np.array(labels)))
    # Define a DataLoader
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    epochs = 300
    for epoch in tqdm(range(epochs)):
        for i, (inputs, targets) in enumerate(train_loader):
            outputs = model((inputs.float()))
            loss = criterion(outputs.squeeze(), (targets.float()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        ###print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, loss.item()))

    save_path = model_dir + "combine_model.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), model_dir + "combine_model.pt")
    torch.save(model.sub_net.state_dict(), model_dir + 'sub_net.pt')

class aiwei23_WatermarkLogitsProcessor(LogitsProcessor):
    """
                Class for injecting watermarks in a sequence of tokens.

                Args:
                    bit_number: The number of encoding bits for the model (>log(len(tokenizer))).
                    sample_number: The number of generated samples for training the watermark generator.
                    data_dir: The path of generated samples for training the watermark generator.
                    window_size: The window size of the method.
                    model_dir: The path of trained watermark generator model.
                    layers: The number of layers of the watermark generator model.
                    llm_name: The LLM we used for generating the text.
            """
    def __init__(self, vocab, delta, model, window_size, cache, bit_number, beam_size, llm_name):
        self.vocab = vocab
        self.delta = delta
        self.model = model
        self.window_size = window_size
        self.cache = cache
        self.bit_number = bit_number
        self.llm_name = llm_name
        if beam_size > 0:
            self.beam_size = beam_size
            self.mode = "beam"
        else:
            self.mode = "sample"

    def _get_greenlist_ids(self, input_ids, scores):
        greenlist_ids = []
        # Get the last 'window_size - 1' items from input_ids
        last_nums = input_ids[-(self.window_size - 1):] if self.window_size - 1 > 0 else []
        if self.mode == "sample":
            _, candidate_tokens = torch.topk(input=scores, k=20, largest=True, sorted=False)
        else:
            # Get the score at index 'beam_size'
            threshold_score = torch.topk(input=scores, k=self.beam_size, largest=True, sorted=False)[0][-1]

            # Get all indices where score is greater than 'score - delta'
            candidate_tokens = (scores >= (threshold_score - self.delta)).nonzero(as_tuple=True)[0]

        for v in candidate_tokens:
            # Append the current number to the list
            pair = list(last_nums) + [v]
            merged_tuple = tuple(pair)
            bin_list = [int_to_bin_list(num, self.bit_number) for num in pair]

            # load & update cache
            if merged_tuple in self.cache:
                result = self.cache[merged_tuple]
            else:
                result = get_value(torch.FloatTensor(bin_list).unsqueeze(0), self.model)
                self.cache[merged_tuple] = result
            if result:
                greenlist_ids.append(int(v))

        return greenlist_ids

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor,
                               greenlist_bias: float) -> torch.Tensor:
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        # if the length of input_id < self.window_size - 1, there is no need to add bias
        if input_ids.shape[-1] < self.window_size - 1:
            if self.llm_name == "gpt2":
                for b_idx in range(input_ids.shape[0]):
                    scores[b_idx][50256] = -10000
            elif self.llm_name == "opt-6.7b":
                for b_idx in range(input_ids.shape[0]):
                    scores[b_idx][2] = -10000
            elif self.llm_name == "llama-7b":
                for b_idx in range(input_ids.shape[0]):
                    scores[b_idx][1] = -10000
            return scores

        green_tokens_mask = torch.zeros_like(scores)
        for b_idx in range(input_ids.shape[0]):
            greenlist_ids = self._get_greenlist_ids(input_ids[b_idx], scores=scores[b_idx])
            green_tokens_mask[b_idx][greenlist_ids] = 1
        green_tokens_mask = green_tokens_mask.bool()

        scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.delta)

        if self.llm_name == "gpt2":
            for b_idx in range(input_ids.shape[0]):
                scores[b_idx][50256] = -10000
        elif self.llm_name == "opt-6.7b":
            for b_idx in range(input_ids.shape[0]):
                scores[b_idx][2] = -10000
        elif self.llm_name == "llama-7b":
            for b_idx in range(input_ids.shape[0]):
                scores[b_idx][1] = -10000

        return scores


class aiwei23_WatermarkDetector:
    def __init__(
            self,
            bit_number: int = 8,
            window_size: int = 5,
            layers: int = 3,
            gamma: float = 0.5,
            delta: float = 2.0,
            model_dir: str = None,
            beam_size: int = 0,
            llm_name = 'gpt2',
            data_dir = './data',
            z_value = 4
    ):
        # watermarking parameters
        self.model_dir = model_dir
        self.bit_number = bit_number
        self.vocab = list(range(1, 2 ** bit_number - 1))
        self.vocab_size = len(self.vocab)
        self.gamma = gamma
        self.min_prefix_len = window_size - 1
        self.window_size = window_size
        self.model = get_model(bit_number, window_size, model_dir+"combine_model.pt", layers)
        self.cache = {}
        self.delta = delta
        self.beam_size = beam_size
        self.llm_name = llm_name
        self.data_dir = data_dir
        self.z_value = z_value
        self.layers = layers

    def _get_cache(self):
        return self.cache

    def random_sample(self, input_ids, is_green):
        # Get the last 'window_size - 1' items from input_ids
        last_nums = input_ids[-(self.window_size - 1):] if self.window_size - 1 > 0 else []
        while True:
            number = random.choice(self.vocab)
            # Append the new random number to the list
            pair = list(last_nums) + [number]
            merged_tuple = tuple(pair)
            bin_list = [int_to_bin_list(num, self.bit_number) for num in pair]

            if merged_tuple in self.cache:
                result = self.cache[merged_tuple]
            else:
                result = get_value(torch.FloatTensor(bin_list).unsqueeze(0), self.model)
                self.cache[merged_tuple] = result

            if is_green and result:
                return number

            elif not is_green and not result:
                return number

    def judge_green(self, input_ids, current_number):
        # Get the last 'window_size - 1' items from input_ids
        last_nums = input_ids[-(self.window_size - 1):] if self.window_size - 1 > 0 else []
        # Append the current number to the list
        pair = list(last_nums) + [current_number]
        merged_tuple = tuple(pair)
        bin_list = [int_to_bin_list(num, self.bit_number) for num in pair]
        # merged_list = sum(bin_list, [])

        # load & update cache
        if merged_tuple in self.cache:
            result = self.cache[merged_tuple]
        else:
            result = get_value(torch.FloatTensor(bin_list).unsqueeze(0), self.model)
            self.cache[merged_tuple] = result

        return result

    def green_token_mask_and_stats(self, input_ids: torch.Tensor):
        mask_list = []
        green_token_count = 0
        for idx in range(self.min_prefix_len, len(input_ids)):
            curr_token = input_ids[idx]
            if self.judge_green(input_ids[:idx], curr_token):
                mask_list.append(True)
                green_token_count += 1
            else:
                mask_list.append(False)
        num_tokens_scored = len(input_ids) - self.min_prefix_len
        z_score = self._compute_z_score(green_token_count, num_tokens_scored)
        return mask_list, green_token_count, z_score

    def _compute_z_score(self, observed_count, T):
        # count refers to number of green tokens, T is total number of tokens
        sigma = 0.01
        expected_count = self.gamma
        numer = observed_count - expected_count * T
        denom = sqrt(T * expected_count * (1 - expected_count) + sigma * sigma * T)
        z = numer / denom
        return z

    def generate_list_with_green_ratio(self, length: int, green_ratio: float):

        token_list = random.sample(self.vocab, self.window_size - 1) if self.window_size - 1 > 0 else random.sample(
            self.vocab, 1)
        is_green = []

        while len(token_list) < length:
            green = 1 if random.random() < green_ratio else 0
            if green:
                token = self.random_sample(torch.LongTensor(token_list), True)
                token_list.append(token)
                is_green.append(1)
            else:
                token = self.random_sample(torch.LongTensor(token_list), False)
                token_list.append(token)
                is_green.append(0)

        # loop
        is_green_append = []
        for i in range(0, self.window_size - 1):
            tail_slice = token_list[-(self.window_size - 1 - i):]
            head_slice = token_list[:i]
            input_slice = tail_slice + head_slice
            is_green_append.append(self.judge_green(input_slice, token_list[i]))

        is_green = is_green_append + is_green

        return token_list, is_green

    def generate_and_save_train_data(self, num_samples):
        print("Generating train data for detector.")
        train_data = []
        for _ in tqdm(range(num_samples)):
            length = 200
            green_ratio = random.random()
            token_list, is_green = self.generate_list_with_green_ratio(length, green_ratio)
            _, _, z_score = self.green_token_mask_and_stats(torch.tensor(token_list))
            train_data.append((tuple(token_list), tuple(is_green), z_score))

        train_data = list(set(train_data))

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        with open(os.path.join(self.data_dir, 'train_data.jsonl'), 'w') as f:
            for item in train_data:
                json.dump(
                    {"Input": [int(i) for i in item[0]], "Tag": [int(i) for i in item[1]], "Output": float(item[2])}, f)
                f.write('\n')

    def generate_and_save_test_data(self, dataset_name, sampling_temp, max_new_tokens):
        """Instatiate the WatermarkLogitsProcessor according to the watermark parameters
        and generate watermarked text by passing it to the generate method of the model
        as a logits processor. """

        print("loading llm...")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Generating test data for detector.")
        if self.llm_name == "gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            model = model.to(device)
        elif self.llm_name == "opt-6.7b":
            tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b", use_fast=False)
            model = AutoModelForCausalLM.from_pretrained("facebook/opt-6.7b", torch_dtype=torch.float16).cuda()
            model = model.to(device)
        elif self.llm_name == "llama-7b":
            tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
            model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf", device_map='auto')

        watermark_processor = aiwei23_WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                       delta=self.delta,
                                                       model=self.model,
                                                       window_size=self.window_size,
                                                       cache=self.cache,
                                                       bit_number=self.bit_number,
                                                       beam_size=self.beam_size,
                                                       llm_name=self.llm_name)
        custom_processor = CustomLogitsProcessor(llm_name=self.llm_name)

        gen_kwargs = dict(max_new_tokens=max_new_tokens)

        if self.beam_size == 0:
            gen_kwargs.update(dict(
                do_sample=True,
                top_k=20,
                temperature=sampling_temp
            ))
        else:
            gen_kwargs.update(dict(
                num_beams=self.beam_size
            ))

        print(gen_kwargs)

        generate_with_watermark = partial(
            model.generate,
            logits_processor=LogitsProcessorList([watermark_processor]),
            no_repeat_ngram_size=4,
            **gen_kwargs
        )

        generate_without_watermark = partial(
            model.generate,
            logits_processor=LogitsProcessorList([custom_processor]),
            **gen_kwargs
        )

        decoded_output_with_watermark = []
        decoded_output_without_watermark = []

        print("dataset")
        print(dataset_name)

        # load dataset
        print("loading dataset...")
        if dataset_name == "c4":
            with open("./original_data/c4_validation.json", encoding="utf-8") as f1:
                lines = f1.readlines()
        elif dataset_name == "dbpedia":
            with open("./original_data/dbpedia_validation.json", encoding="utf-8") as f1:
                lines = f1.readlines()

        idx = 1
        for line in lines:
            try:
                if idx > 500:  # you can change it
                    break
                data = json.loads(line)
                text = data['text']
                text_tokenized = (tokenizer(text, return_tensors="pt", add_special_tokens=True)).to(device)
                prompt_length = 30
                if text_tokenized["input_ids"].shape[-1] < 230:
                    continue

                prompt = {}
                prompt["input_ids"] = text_tokenized["input_ids"][:, : prompt_length]
                prompt["attention_mask"] = text_tokenized["attention_mask"][:, : prompt_length]

                print("generate with watermark...")
                output_with_watermark = generate_with_watermark(**prompt)
                output_with_watermark = output_with_watermark[:, prompt["input_ids"].shape[-1]:]

                print("get unwatermarked text...")
                output_without_watermark = text_tokenized["input_ids"][:, prompt_length:prompt_length + 200]

                _, _, z_score = self.green_token_mask_and_stats(output_with_watermark.squeeze(0))
                decoded_output_with_watermark.append(
                    {"Input": tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0], "Tag": 1,
                     "Z-score": z_score})
                _, _, z_score = self.green_token_mask_and_stats(output_without_watermark.squeeze(0))
                decoded_output_without_watermark.append(
                    {"Input": tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0], "Tag": 0,
                     "Z-score": z_score})

                print(idx)
                idx += 1

            except StopIteration:
                break

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        with open(os.path.join(self.data_dir, 'test_data.jsonl'), 'w') as f:
            for item in decoded_output_with_watermark:
                json.dump({"Input": item["Input"], "Tag": item["Tag"], "Z-score": item["Z-score"]}, f)
                f.write('\n')
            for item in decoded_output_without_watermark:
                json.dump({"Input": item["Input"], "Tag": item["Tag"], "Z-score": item["Z-score"]}, f)
                f.write('\n')

        return watermark_processor

    def train_model(self):
        # Prepare data
        model_file = self.model_dir + "sub_net.pt"
        output_model_dir = "./model/"
        train_data = prepare_data(os.path.join(self.data_dir, 'train_data.jsonl'), train_or_test="train", bit=self.bit_number,
                                  z_value=self.z_value, llm_name=self.llm_name)
        test_data = prepare_data(os.path.join(self.data_dir, 'test_data.jsonl'), train_or_test="test", bit=self.bit_number,
                                 z_value=self.z_value, llm_name=self.llm_name)

        train_dataset = Seq2SeqDataset(train_data)
        test_dataset = Seq2SeqDataset(test_data)

        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=train_collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=32, collate_fn=test_collate_fn)

        # Initialize model and optimizer
        pretrained_dict = torch.load(model_file)
        model = TransformerClassifier(self.bit_number, self.layers, 64, 128)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model_dict = model.binary_classifier.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.binary_classifier.load_state_dict(model_dict, strict=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

        # Define the loss function
        loss_fn = torch.nn.BCELoss()

        for param in model.binary_classifier.parameters():
            param.requires_grad = False

        print("private detector:")
        # save the average acc, tpr, fpr, tnr, fnr of the last 5 epochs
        acc_avg, tpr_avg, fpr_avg, tnr_avg, fnr_avg, f1_avg = 0, 0, 0, 0, 0, 0
        # Train and evaluate
        epochs = 80
        for epoch in tqdm(range(epochs)):
            model.train()
            train_losses = []
            correct = 0
            total = 0
            for inputs, targets in train_dataloader:
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = model((inputs.float()).to(device))
                outputs = outputs.reshape([-1])
                loss = loss_fn(outputs, (targets.float()))
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

                # calculate accuracy
                predicted = (outputs.data > 0.5).float()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            train_accuracy = 100 * correct / total

            model.eval()
            test_losses = []
            correct, total, tp, fp, fn, tn = 0, 0, 0, 0, 0, 0
            with torch.no_grad():
                for inputs, targets, z_score in test_dataloader:
                    outputs = model((inputs.float()).to(device)).to(device)
                    targets = targets.to(device)
                    outputs = outputs.reshape([-1])
                    loss = loss_fn(outputs, targets.float())
                    test_losses.append(loss.item())

                    # calculate acc, tp, fp, fn, tn, f1
                    predicted = (outputs.data > 0.5).int()
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                    tp += (predicted & targets).sum().item()
                    fp += (predicted & (~(targets.bool()))).sum().item()
                    fn += ((~predicted) & targets).sum().item()
                    tn += ((~predicted) & (~(targets.bool()))).sum().item()

            test_accuracy = 100 * correct / total
            test_tpr = 100 * tp / (tp + fn)
            test_fpr = 100 * fp / (fp + tn)
            test_tnr = 100 * tn / (fp + tn)
            test_fnr = 100 * fn / (tp + fn)
            test_f1 = 100 * 2 * tp / (2 * tp + fn + fp)

            ###print(f'Epoch: {epoch}, Train Loss: {sum(train_losses) / len(train_losses)}, Train Accuracy: {train_accuracy}%, Test Loss: {sum(test_losses) / len(test_losses)}, Test Accuracy: {test_accuracy}%, Test TPR: {test_tpr}%, Test FPR: {test_fpr}%, Test TNR: {test_tnr}%, Test FNR: {test_fnr}%, Test F1: {test_f1}%')

            # calculate the average acc, tpr, fpr, tnr, fnr, f1 of the last 5 epochs
            if epochs - 5 <= epoch < epochs:
                acc_avg += test_accuracy
                tpr_avg += test_tpr
                fpr_avg += test_fpr
                tnr_avg += test_tnr
                fnr_avg += test_fnr
                f1_avg += test_f1

        acc_avg /= 5
        tpr_avg /= 5
        fpr_avg /= 5
        tnr_avg /= 5
        fnr_avg /= 5
        f1_avg /= 5

        os.makedirs(os.path.dirname(output_model_dir + "new.pt"), exist_ok=True)
        torch.save(model.binary_classifier.state_dict(), output_model_dir + "new.pt")
        print(
            f'Test Accuracy: {acc_avg}%, Test TPR: {tpr_avg}%, Test FPR: {fpr_avg}%, Test TNR: {tnr_avg}%, Test FNR: {fnr_avg}%, Test F1: {f1_avg}%')

        print("public detector:")
        corr_num, tot_num, tp, fp, fn, tn = 0, 0, 0, 0, 0, 0
        with open(os.path.join(self.data_dir, 'test_data.jsonl'), 'r') as f:
            for line in f:
                tot_num += 1
                json_obj = json.loads(line)
                label = json_obj['Tag']
                z_score = json_obj['Z-score']
                predicted = (z_score > self.z_value)
                if predicted == label:
                    corr_num += 1
                if predicted == 1 and label == 1:
                    tp += 1
                if predicted == 1 and label == 0:
                    fp += 1
                if predicted == 0 and label == 1:
                    fn += 1
                if predicted == 0 and label == 0:
                    tn += 1
        test_accuracy = 100 * corr_num / tot_num
        test_tpr = 100 * tp / (tp + fn)
        test_fpr = 100 * fp / (fp + tn)
        test_tnr = 100 * tn / (fp + tn)
        test_fnr = 100 * fn / (tp + fn)
        test_f1 = 100 * 2 * tp / (2 * tp + fn + fp)
        print(
            f'Test Accuracy: {test_accuracy}%, Test TPR: {test_tpr}%, Test FPR: {test_fpr}%, Test TNR: {test_tnr}%, Test FNR: {test_fnr}%, Test F1: {test_f1}%')

class CustomLogitsProcessor(LogitsProcessor):

    def __init__(self, llm_name):
        super().__init__()
        self.llm_name = llm_name

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.llm_name == "gpt2":
            for b_idx in range(input_ids.shape[0]):
                scores[b_idx][50256] = -10000
        elif self.llm_name == "opt-6.7b":
            for b_idx in range(input_ids.shape[0]):
                scores[b_idx][2] = -10000
        elif self.llm_name == "llama-7b":
            for b_idx in range(input_ids.shape[0]):
                scores[b_idx][1] = -10000
        return scores

if __name__ == "__main__":
    bit_number = 16
    layers = 5
    sample_number = 2000
    window_size = 3
    llm_name = "gpt2"
    gamma = 0.5
    delta = 2.0
    model_dir = "./model/"
    beam_size = 0
    data_dir = "./data"
    z_value = 4
    num_samples = 10000
    dataset_name = "c4"
    sampling_temp = 0.7
    max_new_token = 200

    #prepare_generator(bit_number=bit_number,
    #                  layers=layers,
    #                  sample_number=sample_number,
    #                  window_size=window_size)

    detector = aiwei23_WatermarkDetector(bit_number=bit_number,
                                         window_size=window_size,
                                         layers=layers,
                                         gamma=gamma,
                                         delta=delta,
                                         model_dir=model_dir,
                                         beam_size=beam_size,
                                         llm_name=llm_name,
                                         data_dir=data_dir,
                                         z_value=z_value)

    #detector.generate_and_save_train_data(num_samples=num_samples)
    #watermark_processor = detector.generate_and_save_test_data(dataset_name=dataset_name,
    #                                     sampling_temp=sampling_temp,
    #                                     max_new_tokens=max_new_token)
    detector.train_model()












