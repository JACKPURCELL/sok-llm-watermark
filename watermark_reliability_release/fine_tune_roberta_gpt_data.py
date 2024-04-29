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
import random


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
parser.add_argument('--method_name', type=str, default="john23")
args = parser.parse_args()

print(args.gpus)
num_gpus = 1



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
original_detector_predictions = []
w_wm_output_attacked = []
client = OpenAI(api_key="api_key")




# Load threshold
thresholds = {
   "john23": 2.358025378986253,
   "xuandong23b": 2.545584412271571,
   "aiwei23": 24.998546600341797,
   "rohith23": 1.8526251316070557,
   "xiaoniu23": 0.00,
   "lean23": 0.984638512134552,
   "scott22": 0.17697394677108003,
   "aiwei23b": 0.2496753585975497
}
threshold = thresholds[args.method_name]
baseline_zero_test = []

i = 0
with open('/home/jkl6486/sok-llm-watermark/runs/token_200/roberta_data/gpt_saved_no_prompt.jsonl', 'r') as file:
   for line in file:
       data = json.loads(line)
       if " UNABLE TO COMPLETE REQUEST BECAUSE: length" not in data["output"]:
         generations.append(data["output"])
         labels.append(data["label"])
         i += 1
       if i == 700:
          break

print(str(i) + " 0 samples loaded")

i2 = 0
with open(args.train_path, "r") as f:
   for line in f:
       data = json.loads(line)
       generations.append(data["w_wm_output"])
       labels.append(1)

      
       original_detector_predictions.append(1 if data["w_wm_output_attacked_z_score"] > threshold else 0)
       w_wm_output_attacked.append(data["w_wm_output_attacked"])
       i2 += 1
       if i2 == i:
          break
       
print(str(i2) + " 1 samples loaded")



# with open('/home/jkl6486/sok-llm-watermark/runs/token_200/roberta_data/gpt_saved_llama.jsonl', 'w') as file:
#   for idx in range(len(generations)):
#       item = {"output": generations[idx], "label": labels[idx]}
#       file.write(json.dumps(item) + '\n')







### Comment when test on 0 only
### Shuffle data
random.seed(123)
data_list = list(zip(generations, labels))
random.shuffle(data_list)
generations, labels = zip(*data_list)
generations = list(generations)
labels = list(labels)

train_generations = None
train_labels = None
test_generations = None
test_labels = None

### Split data
amount = int(len(generations))
train_generations = generations[:int(amount*0.8)]
train_labels = labels[:int(amount*0.8)]
test_generations = generations[int(amount*0.8):]
test_labels = labels[int(amount*0.8):]
print("Start training!!!!!!!!!!!!!!!!!!!!")




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
                               evaluation_strategy="no",
                               disable_tqdm=True)
if train_generations is not None:
   train_dataset=dataset(train_generations, train_labels)
   eval_dataset=dataset(test_generations, test_labels)
else:
   train_dataset=dataset(generations, labels)
   eval_dataset=dataset(generations, labels)


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
   res = trainer.predict(dataset(w_wm_output_attacked, [1]*len(w_wm_output_attacked)))
   #res = trainer.predict(dataset(w_wm_output_attacked, [1]*len(w_wm_output_attacked)))
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













