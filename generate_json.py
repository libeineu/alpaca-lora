import argparse
from tqdm import tqdm, trange
from collections import defaultdict

import sys
import re
from tqdm import tqdm
import json

parser = argparse.ArgumentParser(description="src or tgt")
parser.add_argument('--src', '-s', required=True)
parser.add_argument('--tgt', '-t', required=True)
parser.add_argument('--output', '-o', required=True)


args = parser.parse_args()

with open(args.src, 'r') as fr, open(args.tgt, 'r') as tgt, open(args.output, 'w') as out:
    tgt_json = json.load(tgt)
    out_list = []
    for json_item, src_line in zip(tgt_json, fr.readlines()):
        out_list.append({
            "instruction": json_item["instruction"],
            "input": json_item["input"],
            "output": src_line.strip()
        })
    json.dump(out_list, out, ensure_ascii=False, indent=4)