from transformers import AutoTokenizer,LlamaTokenizer

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
tokenizerb = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')


# import argparse


# parser = argparse.ArgumentParser(description="A minimum working example of applying the watermark to any LLM that supports the huggingface 🤗 `generate` API")

# parser.add_argument(
#     "--watermark",
#     type=str,
#     default='xuandong23b',
#     help="Select the watermark type",
# )
# args = parser.parse_args()
# if args.watermark == "xuandong23b":
#     parser.add_argument("--fraction", type=float, default=0.5)
#     parser.add_argument("--strength", type=float, default=2.0)
#     parser.add_argument("--wm_key", type=int, default=0)
    
# args = parser.parse_args()

print("aaa")

