import math

from torch import nn, Tensor
from transformers import BertTokenizer, BertConfig, BertModel, AutoTokenizer, LlamaModel
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import torch
from datetime import datetime
from loss import custom_loss
from transformers import BertForSequenceClassification
from transformers.models.bert.modeling_bert import BertEncoder
#from transformers.models.llama.modeling_llama import Llama
from typing import Any
import random
import hashlib
from random import Random

def train(model,
          train_data,
          epoch_num=5,
          batch_size=10,
          learning_rate=3e-5,
          save_path="./data/BERT_Fine_tuning.pth"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data_size = 104743
    batch_num = int(data_size / batch_size)  ### Only valid for QNLI. QNLI dataset does not support return length by len(dataloader)
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)
    ### Record training
    f = open("./data/result.txt", "a")
    f.write(str(datetime.now()) + " \n")
    f.write("--------Training-------- \n")
    ### Start training
    for epoch in range(epoch_num):
        print("--------Epoch " + str(epoch + 1) + " start----------")
        epoch_loss = 0.
        for index, data in enumerate(tqdm(dataloader, total=batch_num)):
            # Every data instance is an input + label pair
            labels, sentence_1, sentence_2 = data
            # Tokenize input sentences (sentence pair)
            inputs = tokenizer(sentence_1, sentence_2, return_tensors="pt", padding="max_length", truncation=True)
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Zero your gradients for every batch!
            optimizer.zero_grad()
            # Make predictions for this batch
            outputs = model(inputs)
            # Compute the loss and its gradients
            loss = loss_fn(outputs["logits"], labels)
            epoch_loss += loss
            loss.backward()
            # Adjust learning weights
            optimizer.step()
        epoch_loss /= batch_num
        print("Epoch {} loss = {}".format(epoch + 1, epoch_loss))
        f.write("Epoch {} loss = {} \n".format(epoch + 1, epoch_loss))
    ### Save the trained model into file
    torch.save(model.state_dict(), save_path)
    return

def train_student(student_model,
                  teacher_model,
                  train_data,
                  epoch_num=5,
                  batch_size=10,
                  learning_rate=3e-5,
                  temperature=1,
                  save_path="./data/BERT_student.pth"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    student_model.to(device)
    ### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ###
    for para in teacher_model.parameters():
        para.requires_grad=False
    teacher_model.eval()
    teacher_model.to(device)
    data_size = 104743
    batch_num = int(data_size / batch_size)  ### Only valid for QNLI. QNLI dataset does not support return length by len(dataloader)
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=True)
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)
    loss_fn = custom_loss(temperature=temperature)
    loss_fn.to(device)
    ### Record training
    f = open("./data/result_student.txt", "a")
    f.write(str(datetime.now()) + " \n")
    f.write("--------Training-------- \n")
    ### Start training
    for epoch in range(epoch_num):
        print("--------Epoch " + str(epoch + 1) + " start----------")
        epoch_loss = 0.
        for index, data in enumerate(tqdm(dataloader, total=batch_num)):
            # Every data instance is an input + label pair
            labels, sentence_1, sentence_2 = data
            # Tokenize input sentences (sentence pair)
            inputs = tokenizer(sentence_1, sentence_2, return_tensors="pt", padding="max_length", truncation=True)
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Zero your gradients for every batch!
            optimizer.zero_grad()
            # Make predictions for this batch
            student_outputs = student_model(inputs)
            teacher_outputs = teacher_model(inputs)
            teacher_outputs["logits"] = teacher_outputs["logits"].to(device)
            student_outputs["logits"] = student_outputs["logits"].to(device)
            # Compute the loss and its gradients
            loss = loss_fn(teacher_outputs["logits"], student_outputs["logits"], labels)
            epoch_loss += loss
            loss.backward()
            # Adjust learning weights
            optimizer.step()
        epoch_loss /= batch_num
        print("Epoch {} loss = {}".format(epoch + 1, epoch_loss))
        f.write("Epoch {} loss = {} \n".format(epoch + 1, epoch_loss))
    ### Save the trained model into file
    torch.save(student_model.state_dict(), save_path)
    return

def evaluate(model,
             test_data):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    data_size = 5463.0
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)
    batch_size = 10
    dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)
    evaluate_loss = 0.
    correct_num = 0
    f = open("./data/result.txt", "a")
    f.write(str(datetime.now()) + " \n")
    f.write("--------Testing--------")
    batch_num = int(data_size/batch_size)
    with torch.no_grad():
        for index, data in enumerate(tqdm(dataloader, total=batch_num)):
            labels, sentence_1, sentence_2 = data
            inputs = tokenizer(sentence_1, sentence_2, return_tensors="pt", padding="max_length", truncation=True)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs["logits"], labels)
            evaluate_loss += loss
            ### Calculate accuracy
            output_labels = get_labels(outputs["logits"])
            for i in range(len(output_labels)):
                if output_labels[i] == labels[i]:
                    correct_num += 1
    accuracy = correct_num/data_size   ### Calculate accuracy
    evaluate_loss /= batch_num
    print("Loss on test set is {}".format(evaluate_loss))
    f.write("Loss on test set is {} \n".format(evaluate_loss))
    print("Accuracy on test set is {}".format(accuracy))
    f.write("Accuracy on test set is {}".format(accuracy))
    return



def get_labels(logit: Tensor):
    output = logit.softmax(1)
    output_labels = []
    for o in output:
        if o[0] > o[1]:
            output_labels.append(0)
        else:
            output_labels.append(1)
    return output_labels


### Function for distill teacher's weight into student's weight
def distill_weights(
    teacher,
    student,
) -> None:
    """
    Recursively copies the weights of the (teacher) to the (student).
    This function is meant to be first called on a BERT model, but is then called on every children of that model recursively.
    The only part that's not fully copied is the encoder, of which only half is copied.
    """
    # If the part is an entire RoBERTa model or a RobertaFor..., unpack and iterate
    if isinstance(teacher, LlamaModel):
        for teacher_part, student_part in zip(teacher.children(), student.children()):
            distill_weights(teacher_part, student_part)
    # Else if the part is an encoder, copy one out of every layer
    elif isinstance(teacher, LlamaDecoderLayer):
            teacher_encoding_layers = [layer for layer in next(teacher.children())]
            student_encoding_layers = [layer for layer in next(student.children())]
            for i in range(len(student_encoding_layers)):
                student_encoding_layers[i].load_state_dict(teacher_encoding_layers[2*i].state_dict())
    else:
        student.load_state_dict(teacher.state_dict())

### Function for initialize student model
def ini_student(teacher : BertModel) -> nn.Module:
    # Get teacher configuration as a dictionnary
    configuration = teacher.config.to_dict()
    # Half the number of hidden layer
    configuration['num_hidden_layers'] //= 2
    # Convert the dictionnary to the student configuration
    configuration = BertConfig.from_dict(configuration)
    # Create uninitialized student model
    student = type(teacher)(configuration)

    # If want to visualize teacher and student model, uncomment this block
    print("-----------------------------")
    visualize_children(teacher)
    print("-----------------------------")
    visualize_children(student)

    # Initialize the student's weights
    distill_weights(teacher=teacher, student=student)
    # Return the student model

    return student

def visualize_children(
    object : Any,
    level : int = 0,
) -> None:
    """
    Prints the children of (object) and their children too, if there are any.
    Uses the current depth (level) to print things in a ordonnate manner.
    """
    print(f"{'   ' * level}{level}- {type(object).__name__}")
    try:
        for child in object.children():
            visualize_children(child, level + 1)
    except:
        pass

def check_wm(output):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", trust_remote_code=True)
    embedding_list = tokenizer.encode(output)
    count = 0
    green_count = 0
    for token_idx in range(1, len(embedding_list)-1):
        h = hashlib.sha256(str(embedding_list[token_idx]).encode()).hexdigest()
        r_session = Random(h)
        idx_list = r_session.sample(range(0,32000), 16000)
        next_word = embedding_list[token_idx+1]
        if next_word in idx_list:
            green_count += 1
        count += 1
    T = count
    z = 2*(green_count-T/2)/math.sqrt(T)
    return z, green_count/count




if __name__ == "__main__":
    output = "Hello here is a dog putting into a happy face apple company?"
    acc = check_wm(output)
    print(acc)