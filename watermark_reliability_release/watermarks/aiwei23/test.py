from transformers import AutoTokenizer

# 加载 Qwen2.5-14B-Instruct 的 tokenizer
model_name = "Qwen/Qwen2.5-14B-Instruct" 
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 从 tokenizer 中获取 EOS token
eos_token = tokenizer.eos_token
eos_token_id = tokenizer.eos_token_id

# 打印结果
print(f"Qwen2.5 的 EOS token: {eos_token}")
print(f"Qwen2.5 的 EOS token ID: {eos_token_id}")