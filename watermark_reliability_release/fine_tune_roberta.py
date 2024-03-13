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

# load arguments
parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str, default="/home/jkl6486/sok-llm-watermark/runs/token_200/john23/c4/llama/dipper_l40_o0/gen_table_w_metrics.jsonl")
parser.add_argument('--output_path', type=str, default="/home/jkl6486/sok-llm-watermark/runs/token_200/john23/c4/llama/dipper_l40_o0/roberta_finetuned2")
parser.add_argument('--num_epochs', type=int, default=25)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--gpus', type=str, default="0,1,2,3")
parser.add_argument('--train', type=bool, default=True)
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
with open(args.train_path, "r") as f:
	for line in f:
		data = json.loads(line)
		if args.train:
			generations.append(data["truncated_input"]+data["baseline_completion"])
			labels.append(0)
		generations.append(data["truncated_input"]+data["w_wm_output"])
		labels.append(1)
		#original_detector_predictions.append(1 if data["w_wm_output_attacked_prediction"] else 0)
		no_wm_outputs.append(data["truncated_input"]+data["no_wm_output"])
generations = generations[:int(len(generations)*0.8)]
labels = labels[:int(len(labels)*0.8)]
test_generations = generations[int(len(generations)*0.8):]
test_labels = labels[int(len(labels)*0.8):]
test_original_detector_predictions = original_detector_predictions[int(len(original_detector_predictions)*0.8):]


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

trainer = Trainer(model=clf, args=train_args,
				  train_dataset=dataset(generations, labels),
				  eval_dataset=dataset(test_generations, test_labels),
				  tokenizer=tokenizer,
				  compute_metrics=compute_metrics)


if args.train:
    trainer.train()
    trainer.save_model(args.output_path)
	

metrics = trainer.evaluate()
print("Metrics:")
print(metrics)

### predict
# test_generations_new = []
# test_labels_new = []
# if args.train:
# 	for i in range(len(test_generations)):
# 		if test_labels[i] == 0:
# 			test_generations_new.append(test_generations[i])
# 			test_labels_new.append(test_labels[i])
# else:
# 	test_generations_new = test_generations
# 	test_labels_new = test_labels
        
# res = trainer.predict(dataset(test_generations_new, test_labels_new))
# res_labels = res.predictions.argmax(axis=-1)
# acc_count = sum(1 for x, y in zip(res_labels, test_original_detector_predictions) if x == 0 and y == 0)
# print("Accuracy on non-watermarked samples:")
# print(acc_count/sum(res_labels == 0))

# print("END: fine-tuning model")

# res = trainer.predict(dataset(no_wm_outputs, [0]*len(no_wm_outputs)))
# print(res)
# print()










