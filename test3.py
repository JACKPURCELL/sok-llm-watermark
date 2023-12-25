from transformers import AutoTokenizer,LlamaTokenizer

tokenize = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')


print(tokenize('hello world'))
# print(tokenize('hello wordl'))
# print(tokenize('hello word l'))
print(tokenize('hello preferences'))
print(tokenize.decode([2063]))
