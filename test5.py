from transformers import AutoTokenizer,LlamaTokenizer

# tokenize = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
tokenize = AutoTokenizer.from_pretrained('facebook/opt-1.3b')

sentence1 = "He's a tough player. He's physical. He's athletic."
sentence2 = "He is a tough player. he's physical. he's athletic."

print(tokenize(sentence1,return_tensors='pt'))
print(tokenize(sentence2,return_tensors='pt'))