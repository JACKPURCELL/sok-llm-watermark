import json

if __name__ ==  "__main__":
    thresholds = {
    "john23": 2.358025378986253,
    "xuandong23b": 2.545584412271571,
    "aiwei23": 24.998546600341797,
    "rohith23": 1.8526251316070557,
    "xiaoniu23": 0.00,
    "lean23": 0.984638512134552,
    "scott22": 0.17697394677108003,
    "aiwei23b": 0.2496753585975497
    }

    results = {}
    
    for method in ["john23", "rohith23", "xuandong23b", "scott22"]:
        for rep in range(2, 6):
            path =  "/home/ljc/sok-llm-watermark/runs/token_200/" + method + "/c4/opt/dipper_40_rep" + str(rep) + "/escape/gen_table_w_metrics.jsonl"
            with open(path, "r") as f:
                wm_count = 0
                no_wm_count = 0
                for line in f:
                    data = json.loads(line)
                    if data["w_wm_output_attacked_z_score"] > thresholds[method]:
                        wm_count += 1
                    else:
                        no_wm_count += 1
                results[method + "_rep" + str(rep)] = {"wm": wm_count, "no_wm": no_wm_count}
                
    print(results)
    print()
            
    
    
    
    
    
    
    