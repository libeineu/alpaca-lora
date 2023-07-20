"""
Convert alpaca dataset into sharegpt format.

Usage: python3 -m fastchat.data.convert_alpaca --in alpaca_data.json
"""

import argparse
import json
import numpy as np
import random

import argparse
from tqdm import tqdm, trange
from collections import defaultdict

parser = argparse.ArgumentParser(description="src or tgt")
parser.add_argument('--data1', '-i1', required=True)
parser.add_argument('--data2', '-i2', required=True)
parser.add_argument('--output', '-o', required=True)

args = parser.parse_args()

if __name__ == "__main__":
    content = json.load(open(args.data1, "r", encoding="utf-8"))
    # json.load(
    sharegpt_all = []
    for i, c in enumerate(content):
        sharegpt_all.append(c)
    
    content = json.load(open(args.data2, "r", encoding="utf-8"))
    alpaca_all = []
    for i, c in enumerate(content):
        alpaca_all.append(c)

    all_content = []
    all_content += sharegpt_all
    all_content += alpaca_all

    final_index = random.sample(range(len(all_content)), len(all_content))
    final_data = []
    for index in final_index:
        final_data.append(all_content[index])

    print(f"#out: {len(final_data)}")
    data_size = len(final_data)
    json.dump(final_data, open(f"{args.output}.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
