import json

count = 0
with open('/home/ljc/sok-llm-watermark/runs/xuandong23b/dipper.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data['attack_output_result']['prediction'] == True:
            count += 1

print(count)
print(count/1000)