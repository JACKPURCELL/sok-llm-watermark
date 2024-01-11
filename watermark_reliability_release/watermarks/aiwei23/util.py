import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import torch.nn.functional as F
from transformers import GPT2Tokenizer, AutoTokenizer, LlamaTokenizer
import torch.nn as nn
import json


### Util functions
def get_model(input_dim, window_size, model_dir, layers=3):
    model = BinaryClassifier(input_dim, window_size, layers)
    if model_dir is not None:
        model.load_state_dict(torch.load(model_dir))
    return model

def int_to_bin_list(n, length=8):
    bin_str = format(n, 'b').zfill(length)
    return [int(b) for b in bin_str]

def load_data(filepath):
    features = []
    labels = []
    with open(filepath, 'r') as file:
        for line in file:
            entry = json.loads(line)
            features.append(entry['data'])
            labels.append(entry['label'])
    return features, labels

def max_number(bits):
    return (1 << bits) - 1

def get_value(input_x, model):
    output = model(input_x)
    output = (output > 0.5).bool().item()
    return output


def prepare_data(filepath, train_or_test="train", llm_name="gpt2", bit=16, z_value=4):
    data = []
    if train_or_test == "train":
        with open(filepath, 'r') as f:
            for line in f:
                json_obj = json.loads(line)
                inputs = json_obj['Input']
                output = json_obj['Output']
                label = 1 if output > z_value else 0  # binary classification

                inputs_bin = [int_to_bin_list(n, bit) for n in inputs]

                data.append((torch.tensor(inputs_bin), torch.tensor(label)))  # label is a scalar
    else:
        with open(filepath, 'r') as f:
            for line in f:
                json_obj = json.loads(line)
                inputs = json_obj['Input']
                label = json_obj['Tag']
                z_score = json_obj['Z-score']

                if llm_name == "gpt2":
                    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                    inputs = tokenizer(inputs, return_tensors="pt", add_special_tokens=True)
                elif llm_name == "opt-1.3b":
                    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", use_fast=False)
                    inputs = tokenizer(inputs, return_tensors="pt", add_special_tokens=True)
                elif llm_name == "llama-7b":
                    tokenizer = AutoTokenizer.from_pretrained(
                            "meta-llama/Llama-2-7b-chat-hf", padding_side="left",return_token_type_ids=False
                        )
        
                    inputs = tokenizer(inputs, return_tensors="pt", add_special_tokens=True)

                inputs_bin = [int_to_bin_list(n, bit) for n in inputs["input_ids"].squeeze()[1:]]

                data.append((torch.tensor(inputs_bin), torch.tensor(label), torch.tensor(z_score)))  # label is a scalar

    return data


def pad_sequence_to_fixed_length(inputs, target_length, padding_value=0):
    padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=padding_value)

    original_length = padded_inputs.shape[1]

    if original_length < target_length:
        # If the original sequence is shorter than the target length, we need to further pad the sequences
        pad_size = (0, 0, 0, target_length - original_length)
        padded_inputs = F.pad(padded_inputs, pad_size, value=padding_value)
    elif original_length > target_length:
        # If the original sequence is longer than the target length, we need to truncate the sequences
        padded_inputs = padded_inputs[:, :target_length, :]
    else:
        # If the original sequence is the same as the target length, just return the original inputs
        padded_inputs = padded_inputs

    return padded_inputs


def train_collate_fn(batch):
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    inputs_padded = pad_sequence_to_fixed_length(inputs, 200)

    return inputs_padded, torch.stack(targets)


def test_collate_fn(batch):
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    z_score = [item[2] for item in batch]

    inputs_padded = pad_sequence_to_fixed_length(inputs, 200)

    return inputs_padded, torch.stack(targets), torch.stack(z_score)



class Seq2SeqDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, window_size, num_layers, hidden_dim=64):
        super(BinaryClassifier, self).__init__()

        # subnet
        self.sub_net = SubNet(input_dim, num_layers, hidden_dim)
        self.window_size = window_size
        self.relu = nn.ReLU()

        # linear layer and sigmoid layer after merging features
        self.combine_layer = nn.Linear(window_size * hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x is expected to be of shape (batch_size, window_size, input_dim)
        batch_size = x.shape[0]
        # Reshape x to (-1, input_dim) so it can be passed through the sub_net in one go
        x = x.view(-1, x.shape[-1])
        sub_net_output = self.sub_net(x)

        # Reshape sub_net_output back to (batch_size, window_size*hidden_dim)
        sub_net_output = sub_net_output.view(batch_size, -1)
        combined_features = self.combine_layer(sub_net_output)
        combined_features = self.relu(combined_features)
        output = self.output_layer(combined_features)
        output = self.sigmoid(output)

        return output

class SubNet(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_dim=64):
        super(SubNet, self).__init__()
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())

        # Output layer
        self.layers.append(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, bit_number, b_layers, input_dim, hidden_dim, num_classes=1, num_layers=2):
        super(TransformerClassifier, self).__init__()
        self.binary_classifier = SubNet(bit_number, b_layers)
        self.classifier = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x1 = x.view(batch_size*seq_len, -1)
        features = self.binary_classifier(x1)
        features = features.view(batch_size, seq_len, -1)  # Ensure LSTM compatible shape
        output, _ = self.classifier(features)
        output = self.fc_hidden(output[:, -1, :])  # Take the last LSTM output for classification
        output = self.dropout(output)
        output = self.sigmoid(output)
        output = self.fc(output)
        output = self.dropout(output)
        output = self.sigmoid(output)
        return output








