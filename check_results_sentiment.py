# import json
# import os

# if __name__ == "__main__":
#     server = "Server 3"
#     attackS = ["ContractionAttack", "copypaste-1-10", "copypaste-3-10", "copypaste-1-25", "copypaste-3-25","dipper_l20_o0", "ExpansionAttack", "LowercaseAttack", "MisspellingAttack", "swap", "synonym-0.4", "TypoAttack"]
#     modelS = ["opt" "llama"]
#     # methodS = ["aiwei23", "john23", "lean23", "rohith23", "xiaoniu23", "xuandong23b","aiwei23b"]
#     methodS = ["john23"]
#     no_sentiment = []
#     for method in methodS:
#         for model in modelS:    
#             file_path = "/home/jkl6486/sok-llm-watermark/runs/token_200/" + method + "/c4/" + model + "/"
#             if os.path.exists(file_path+"gen_table_w_metrics.jsonl") :
#                 with open(file_path+"gen_table_w_metrics.jsonl", 'r') as f:
#                     first_line = f.readline()  # 读取第一行
#                     data = json.loads(first_line)  # 解析JSON数据

#                     if any('BLEU' in key for key in data.keys()):
#                         pass
#                     else:
#                         no_sentiment.append(method)
                            
#     print("Server: "+server)
#     print()
#     print("These methods do not have sentiment results:")
#     for method in no_sentiment:
#         print(method)


import json
import os

if __name__ == "__main__":
    server = "Server 3"
    attackS = ["ContractionAttack", "copypaste-1-10", "copypaste-3-10", "copypaste-1-25", "copypaste-3-25","dipper_l20_o0", "ExpansionAttack", "LowercaseAttack", "MisspellingAttack", "swap", "synonym-0.4", "TypoAttack"]
    modelS = ["opt","llama"]
    # methodS = ["aiwei23", "john23", "lean23", "rohith23", "xiaoniu23", "xuandong23b","aiwei23b"]
    methodS = ["john23"]
    no_sentiment = []
    for method in methodS:
        for model in modelS:
            for attack in attackS:    
                file_path = "/home/jkl6486/sok-llm-watermark/runs/token_200/" + method + "/c4/" + model + "/" +attack+ "/" 
                if os.path.exists(file_path+"gen_table_w_metrics.jsonl") :
                    with open(file_path+"gen_table_w_metrics.jsonl", 'r') as f:
                        first_line = f.readline()  # 读取第一行
                        data = json.loads(first_line)  # 解析JSON数据

                        if any('BLEU' in key for key in data.keys()):
                            pass
                        else:
                            no_sentiment.append(model+"_"+attack)
                            
    print("Server: "+server)
    print()
    print("These methods do not have sentiment results:")
    for attack in no_sentiment:
        print(attack)















