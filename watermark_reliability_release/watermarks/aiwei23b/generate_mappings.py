import random
import os
import json
import argparse

def generate_mapping(size=30000, dimension=300):
    return [random.randint(0, dimension-1) for _ in range(size)]


def main(args):
    length = args["length"]
    mapping = generate_mapping(length, 300)
    
    output_path = os.path.join(args["output_dir"], f"300_mapping_{length}.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(mapping, f, indent=4)
    

if __name__ == '__main__':
    main()