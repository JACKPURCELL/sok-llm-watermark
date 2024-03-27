"""
Finetune RoBERTa detector
"""
import argparse
import os
import torch
import numpy as np
import pandas as pd
import transformers
import json

import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaForSequenceClassification, DataCollatorWithPadding, AutoModelForSequenceClassification

from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import RobertaTokenizer
from transformers import TrainingArguments, Trainer

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from openai import OpenAI

# load arguments
parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str, default="/home/jkl6486/sok-llm-watermark/runs/token_200/john23/c4/opt/back/dipper_l40_o0/gen_table_w_metrics.jsonl")
parser.add_argument('--output_path', type=str, default="/home/jkl6486/sok-llm-watermark/runs/token_200/john23/c4/opt/back/dipper_l40_o0/roberta_finetuned2")
parser.add_argument('--num_epochs', type=int, default=25)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--gpus', type=str, default="0,1,2,3")
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--eva_attack', type=bool, default=True)
parser.add_argument('--model_name', type=str, default="gpt-3.5-turbo")
parser.add_argument('--max_tokens', type=int, default=200)
args = parser.parse_args()

num_gpus = args.gpus.count(",") + 1
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus 

# load model and tokenizer
if args.train:
	
    clf = RobertaForSequenceClassification.from_pretrained("roberta-base",
    														num_labels=2)
else:
    clf = AutoModelForSequenceClassification.from_pretrained(args.output_path)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
device = torch.device("cuda")


# load training dataset
generations = []
labels = []
no_wm_outputs = []
original_detector_predictions = []
w_wm_output_atatcked = []
client = OpenAI(api_key="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")

i = 0
with open(args.train_path, "r") as f:
	for line in f:
		data = json.loads(line)
		#generations.append(data["truncated_input"]+data["baseline_completion"])
		# prompt = data["truncated_input"]
		# completion = client.chat.completions.create(model=args.model_name,
        #                                             messages=[{"role": "system", "content": "You are a helpful assistant."},
        #                                                       {"role": "user", "content": prompt}],
        #                                             max_tokens=args.max_tokens)
		# # set message
		# if completion.choices[0].finish_reason != "stop":
		# 	msg = "UNABLE TO COMPLETE REQUEST BECAUSE: %s" % completion.choices[0].finish_reason
		# else:
		# 	msg = completion.choices[0].message.content
		# generations.append(data["truncated_input"]+" "+msg)
		# labels.append(0)
		# generations.append(data["truncated_input"]+data["w_wm_output"])
		# labels.append(1)

		### Other variables 
		original_detector_predictions.append(1 if data["w_wm_output_attacked_prediction"] else 0)
		no_wm_outputs.append(data["truncated_input"]+data["no_wm_output"])
		w_wm_output_atatcked.append(data["w_wm_output_attacked"])
		i += 1

# with open('/home/jkl6486/sok-llm-watermark/runs/token_200/john23/c4/opt/back/dipper_l40_o0/roberta_finetuned2/gpt_saved.jsonl', 'w') as file:
# 	for idx in range(len(generations)):
# 		item = {"output": generations[idx], "label": labels[idx]}
# 		file.write(json.dumps(item) + '\n')

with open('/home/jkl6486/sok-llm-watermark/runs/token_200/john23/c4/opt/back/dipper_l40_o0/roberta_finetuned2/gpt_saved.jsonl', 'r') as file:
	for line in file:
		data = json.loads(line)
		generations.append(data["output"])
		labels.append(data["label"])



amount = int(len(generations))
train_generations = generations[:int(amount*0.8)]
train_labels = labels[:int(amount*0.8)]
test_generations = generations[int(amount*0.8):]
test_labels = labels[int(amount*0.8):]
test_original_detector_predictions = original_detector_predictions[int(amount*0.8):]
test_w_wm_output_atatcked = w_wm_output_atatcked[int(amount*0.8):]


class dataset(Dataset):
	"""
	Dataset for dataframes
	"""
	def __init__(self, generations, labels):
		self.generations = generations
		self.labels = labels

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):

		text = self.generations[idx]
		label = self.labels[idx]

		tok = tokenizer(text, truncation=True)

		return {**tok, "labels": label}


print("START: fine-tuning model")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc}


train_args = TrainingArguments(output_dir=args.output_path,
							    num_train_epochs=args.num_epochs,
								learning_rate=args.lr,
								per_device_train_batch_size=int(args.batch_size/num_gpus),
								per_device_eval_batch_size=int(args.batch_size/num_gpus),
								weight_decay=0.01,
								warmup_ratio=0.06,
								save_strategy="no",
								logging_steps=1,
								evaluation_strategy="no")

train_dataset=dataset(train_generations, train_labels)
eval_dataset=dataset(test_generations, test_labels)

trainer = Trainer(model=clf, args=train_args,
				  train_dataset=train_dataset,
				  eval_dataset=eval_dataset,
				  tokenizer=tokenizer,
				  compute_metrics=compute_metrics)


if args.train:
	trainer.train()
	trainer.save_model(args.output_path)
	metrics = trainer.evaluate()
	print("Metrics:")
	print(metrics)

if args.eva_attack:
	res = trainer.predict(dataset(w_wm_output_atatcked, [1]*len(w_wm_output_atatcked)))
	res_labels = res.predictions.argmax(axis=-1)
	acc = sum(res_labels == 1)/len(res_labels)
	print("Accuracy on attacked watermarked samples:")
	print(acc)
	g_0_s_0 = 0
	g_1_s_0 = 0
	g_0_s_1 = 0
	g_1_s_1 = 0
	for t in range(len(res_labels)):
		if (res_labels[t] == 1) and (original_detector_predictions[t] == 1):
			g_1_s_1 += 1
		elif(res_labels[t] == 0) and (original_detector_predictions[t] == 1):
			g_0_s_1 += 1
		elif(res_labels[t] == 1) and (original_detector_predictions[t] == 0):
			g_1_s_0 += 1
		else:
			g_0_s_0 += 1
	print("g_0_s_0: ", g_0_s_0)
	print("g_1_s_0: ", g_1_s_0)
	print("g_0_s_1: ", g_0_s_1)
	print("g_1_s_1: ", g_1_s_1)

	print("Number of 0 predictions:")
	print(sum(res_labels == 0))
	print("Number of 1 predictions:")
	print(sum(res_labels == 1))
	acc_count = sum(1 for x, y in zip(res_labels, original_detector_predictions) if x == 0 and y == 0)
	print("Accuracy on non-watermarked samples:")
	print(acc_count/sum(res_labels == 0))





