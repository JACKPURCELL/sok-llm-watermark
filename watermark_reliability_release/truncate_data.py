import json
import argparse
from transformers import AutoTokenizer
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods_name', type=list, default=["john23", "lean23", "rohith23", "xiaoniu23", "xuandong23b", "aiwei23", "scott22"])
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
    for method in args.methods_name:
        try:
            data_path = "/home/jkl6486/sok-llm-watermark/runs/token_200/" + method + "/c4/opt/gen_table_w_metrics.jsonl"
            try:
                os.mkdir("/home/jkl6486/sok-llm-watermark/runs/token_200/" + method + "/c4/opt/truncated")
            except:
                print("!!!")
            for length in range(10, 201, 10):
                with open (data_path, "r") as f:
                    idx = 0
                    for line in f:
                        data = json.loads(line)
                        if data["w_wm_output_prediction"] and data["w_wm_output_length"] > 190:
                        #if data["w_wm_output_z_score"] > 0.2496753585975497 and data["w_wm_output_length"] > 190:
                            modified_data = {}
                            output_path = "/home/jkl6486/sok-llm-watermark/runs/token_200/" + method + "/c4/opt/truncated/gen_table.jsonl"
                            tokenized_input = tokenizer.encode(data["w_wm_output"], add_special_tokens=False)
                            modified_data = data.copy()
                            modified_data["w_wm_output"] = tokenizer.decode(tokenized_input[:length], skip_special_tokens=True)
                            modified_data["w_wm_output_length"] = length
                            with open(output_path, "a") as f2:
                                json.dump(modified_data, f2)
                                f2.write("\n")
                            idx += 1
                        if idx == 100:
                            break
        except:
            print("Method " + method + " failed")






                    
