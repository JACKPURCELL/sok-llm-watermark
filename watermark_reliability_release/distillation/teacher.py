from collections import OrderedDict

import torchtext.datasets
from torch import nn
from transformers import AutoModelForSequenceClassification
import torch
from utility import train, evaluate

class Teacher(nn.Module):
    ### Simply split the data into train and test (no dev)
    ### Here we use dev set as our test set, because test set of QNLI has no label
    ### hard code dataset, num of classification labels in the __init__ function
    def __init__(self):
        super(Teacher, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
    def forward(self, input):
        return self.model(**input)
    def load(self, path):
        #self.model.load_state_dict(torch.load(path))
        state_dict = OrderedDict()
        for k,v in torch.load(path).items():
            k = k[6:len(k)]
            state_dict[k] = v
        self.model.load_state_dict(state_dict)
        return


if __name__ == "__main__":
    teacher_model = Teacher()
    train_data = torchtext.datasets.QNLI(split="train")
    test_data = torchtext.datasets.QNLI(split="dev")
    #train(teacher_model, train_data)
    teacher_model.load("./data/BERT_Fine_tuning.pth")
    print("-------------------Evaluating-------------------")
    evaluate(teacher_model, train_data)
    evaluate(teacher_model, test_data)
    print()