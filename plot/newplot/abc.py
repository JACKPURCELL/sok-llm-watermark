


attacks = ["swap","translation","synonym-0.4", "copypaste-1-10","copypaste-3-10","copypaste-1-25","copypaste-3-25", "ContractionAttack", "ExpansionAttack",  "MisspellingAttack",   "dipper_l20_o0", "dipper_l40_o0",  "dipper_l60_o0",  "dipper_l40_o20", "dipper_l60_o20",     "dipper_l60_o40",    "LowercaseAttack", "TypoAttack"]


for attack in attacks:
 data_list = read_file(f'/home/jkl6486/sok-llm-watermark/runs/token_200/aiwei23/c4/opt/{attack}/gen_table_w_metrics.jsonl')