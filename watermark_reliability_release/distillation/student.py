from collections import OrderedDict

import torch
import torchtext
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader

import utility
from utility import ini_student, train_student, evaluate, visualize_children
from Teacher import Teacher
from transformers import AutoModelForCausalLM, OPTForCausalLM, OPTModel, AutoTokenizer, LogitsProcessorList
from transformers.models.opt.modeling_opt import OPTDecoder, OPTConfig
from torch.nn import ModuleList, CrossEntropyLoss
import sys
sys.path.append("./reference_code/lm-watermarking")
from extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
from sklearn.metrics import jaccard_score

def distill_weights(
    teacher,
    student,
) -> None:
    """
    Recursively copies the weights of the (teacher) to the (student).
    This function is meant to be first called on a BERT model, but is then called on every children of that model recursively.
    The only part that's not fully copied is the encoder, of which only half is copied.
    """

    if isinstance(teacher, OPTForCausalLM) or isinstance(teacher, OPTModel) or isinstance(teacher, OPTDecoder):
        for teacher_part, student_part in zip(teacher.children(), student.children()):
            distill_weights(teacher_part, student_part)

    elif isinstance(teacher, ModuleList):
            teacher_encoding_layers = [layer for layer in teacher.children()]
            student_encoding_layers = [layer for layer in student.children()]
            for i in range(len(student_encoding_layers)):
                student_encoding_layers[i].load_state_dict(teacher_encoding_layers[2*i].state_dict())
    else:
        student.load_state_dict(teacher.state_dict())

### Function for initialize student model
def ini_student(teacher : OPTForCausalLM) -> nn.Module:
    # Get teacher configuration as a dictionnary
    configuration = teacher.config.to_dict()
    # Half the number of hidden layer
    configuration['num_hidden_layers'] //= 2
    # Convert the dictionnary to the student configuration
    configuration = OPTConfig.from_dict(configuration)
    # Create uninitialized student model
    student = type(teacher)(configuration)

    # If want to visualize teacher and student model, uncomment this block
    #print("-----------------------------")
    #visualize_children(teacher)
    #print("-----------------------------")
    #visualize_children(student)

    # Initialize the student's weights
    distill_weights(teacher=teacher, student=student)
    # Return the student model

    return student



class Student(OPTForCausalLM):
    def __init__(self, teacher_model : OPTForCausalLM):
        # Get teacher configuration as a dictionnary
        configuration = teacher_model.config.to_dict()
        # Half the number of hidden layer
        configuration['num_hidden_layers'] //= 2
        configuration = OPTConfig.from_dict(configuration)
        super().__init__(configuration)


    def forward(self, x):
        return self.model(**x)

    def load(self, path):
        state_dict = OrderedDict()
        for k, v in torch.load(path).items():
            k = k[8:len(k)]
            state_dict[k] = v
        self.student.load_state_dict(state_dict)
        return


def train(model,
          train_data,
          epoch_num=5,
          batch_size=1,
          learning_rate=3e-5,
          temperature=1,
          save_path="./data/Llama2_student.pth"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ### Freeze the teacher model to make it run faster ###
    for para in teacher_model.parameters():
        para.requires_grad = False
    teacher_model.eval()
    teacher_model.to(device)
    ### Define tokenizer
    tokenizer = AutoTokenizer.from_pretrained("data/llama", trust_remote_code=True)
    ### Data loader
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=True)
    ### Assign batch size
    data_size = len(train_data)
    batch_num = int(data_size / batch_size)
    ### Optimizer & loss function
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)
    loss_fn = custom_loss(temperature=temperature)
    loss_fn.to(device)
    ### Record training
    f = open("./data/result_student_MMLU.txt", "a")
    f.write(str(datetime.now()) + " \n")
    f.write("--------Training-------- \n")
    ### Start training
    for epoch in range(epoch_num):
        print("--------Epoch " + str(epoch + 1) + " start----------")
        epoch_loss = 0.
        for index, data in enumerate(tqdm(dataloader, total=batch_num)):
            prompt, answer = data
            new_token = ""
            for t in range(len(answer)):
                prompt = prompt + new_token
                input = tokenizer(prompt, return_tensors="pt")
                watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                               gamma=0.25,
                                                               delta=2.0,
                                                               seeding_scheme="selfhash")
                generate_ids = student_model.generate(inputs.input_ids,
                                                      max_length=90,
                                                      logits_processor=LogitsProcessorList([watermark_processor]),
                                                      output_scores=True,
                                                      return_dict_in_generate=True,
                                                      max_new_tokens=1
                                                      )



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

def loss_function(output_logits, target_logits):
    return CrossEntropyLoss(output_logits, target_logits)


def load_data():
    data = load_dataset("cais/mmlu", "all", split="dev")
    dataloader = DataLoader(data, batch_size=1, shuffle=False, drop_last=True)
    for i, d in enumerate(dataloader):
        pass




#def build_dataset(teacher):



if __name__ == "__main__":
    torch.set_default_device("cuda:0")
    load_data()
    #teacher_model = Teacher()
    #teacher_model.load("./data/BERT_Fine_tuning.pth")
    model_path = "facebook/opt-1.3b"
    teacher_model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b")
    #student_model = Student(teacher_model.model)
    teacher_config = teacher_model.config.to_dict()
    teacher_config['num_hidden_layers'] //= 2
    teacher_config = OPTConfig.from_dict(teacher_config)
    student_model = AutoModelForCausalLM.from_config(teacher_config)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompt = "Hi,could you explain what is chatgpt for me? Sure"
    inputs = tokenizer(prompt, return_tensors="pt")
    watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                   gamma=0.25,
                                                   delta=2.0,
                                                   seeding_scheme="selfhash")
    generate_ids = teacher_model.generate(inputs.input_ids,
                                          max_length=90,
                                          logits_processor=LogitsProcessorList([watermark_processor]),
                                          output_scores=True,
                                          return_dict_in_generate=True,
                                          max_new_tokens=90,
                                          num_beams=3,
                                          num_return_sequences=3   ### top K sequences returned from beam search
                                          )


    output = generate_ids["sequences"][:, inputs["input_ids"].shape[-1]:]
    
    output = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    #output = student_model(**inputs)
    #output = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(output)
    #train_data = torchtext.datasets.QNLI(split="train")
    #test_data = torchtext.datasets.QNLI(split="dev")
    #student_model.load("./data/Bert_student.pth")
    #evaluate(teacher_model, train_data)
    #evaluate(teacher_model, test_data)
    #train_student(student_model, teacher_model, train_data, epoch_num=5, learning_rate=3e-5)
    #"./data/BERT_student.pth"
    #evaluate(student_model, test_data)
    #evaluate(student_model, train_data)
    a = jaccard_score("asd","asd")
    print()