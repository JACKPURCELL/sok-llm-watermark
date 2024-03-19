import os

if __name__ == "__main__":
    server = "Server 3"
    attackS = ["ContractionAttack", "copypaste-1-10", "copypaste-3-10", "copypaste-1-25", "copypaste-3-25","dipper_l20_o0", "dipper_l40_o0","ExpansionAttack", "LowercaseAttack", "MisspellingAttack", "swap", "synonym-0.4", "TypoAttack"]
    modelS = ["llama"]
    # methodS = ["john23", "lean23"]
    methodS = ["aiwei23", "john23", "lean23", "rohith23", "xiaoniu23", "xuandong23b","aiwei23b"]
    no_atk = {}
    no_eva = {}
    for method in methodS:
        for model in modelS:    
            for attack in attackS:
                file_path = "~/sok-llm-watermark/runs/token_200/" + method + "/c4/" + model + "/" + attack + "/"
                if not os.path.exists(file_path+"gen_table_attacked.jsonl") or not os.path.exists(file_path+"gen_table_attacked_meta.json"):
                    if method in no_atk:
                        no_atk[method].append(attack)
                    else:
                        no_atk[method] = [attack]
                    ### Add more here
                if not os.path.exists(file_path+"gen_table_w_metrics.jsonl") or not os.path.exists(file_path+"gen_table_w_metrics_meta.json"):
                    if method in no_eva:
                        no_eva[method].append(attack)
                    else:
                        no_eva[method] = [attack]
                    ### Add more here

    print("Server: "+server)
    print()
    print("These methods do not have attack results:")
    for method in no_atk.keys():
        print(method)
        for attack in no_atk[method]:
            print("--- "+attack)
    print()
    print()
    print("These methods do not have evaluation results:")
    for method in no_eva.keys():
        print(method)
        for attack in no_eva[method]:
            print("--- "+attack)
    print()
    





















