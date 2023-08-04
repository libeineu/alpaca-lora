import argparse
import os
import requests
from bs4 import BeautifulSoup

urls = {'llama7b': ['https://huggingface.co/decapoda-research/llama-7b-hf/tree/main', ''],
        'llama13b': ["https://huggingface.co/decapoda-research/llama-13b-hf/tree/main", ''],
        'alpaca-lora': ["https://huggingface.co/tloen/alpaca-lora-7b/tree/main", ''],
        'alpaca7b': ['https://huggingface.co/chavinlo/alpaca-native/tree/main', ''],
        'alpaca13b': ['https://huggingface.co/chavinlo/alpaca-13b/tree/main', ''],
        'vicuna13b_v1.5': ['https://huggingface.co/lmsys/vicuna-13b-v1.5/tree/main', ''],
        'sharedgpt': ['https://huggingface.co/datasets/philschmid/sharegpt-raw/tree/main/sharegpt_90k_raw_dataset', ''],
        'llama30b': ['https://huggingface.co/decapoda-research/llama-30b-hf/tree/main',
                     'https://huggingface.co/api/models/decapoda-research/llama-30b-hf/tree/main?cursor=ZXlKbWFXeGxYMjVoYldVaU9pSndlWFJ2Y21Ob1gyMXZaR1ZzTFRBd01EUTBMVzltTFRBd01EWXhMbUpwYmlKOTo1MA=='],
        'llama65b': ['https://huggingface.co/decapoda-research/llama-65b-hf/tree/main',
                     'https://huggingface.co/api/models/decapoda-research/llama-65b-hf/tree/main?cursor=ZXlKbWFXeGxYMjVoYldVaU9pSndlWFJ2Y21Ob1gyMXZaR1ZzTFRBd01EUTFMVzltTFRBd01EZ3hMbUpwYmlKOTo1MA=='],
        }

args = argparse.ArgumentParser()
args.add_argument('--model', default='llama7b', nargs='?')
args.add_argument('--output-dir', default='llama7b', nargs='?')
parser = args.parse_args()
url = urls[parser.model]
user_name = "v-lbei"

response = requests.get(url[0])
soup = BeautifulSoup(response.content, 'html.parser')
try:
    response_for_implicit = requests.get(url[1])
    json_page = response_for_implicit.json()

except:
    print("加载隐藏页面失败")
# 查找具有特定标题的所有a标签
a_tags = soup.find_all('a', title="Download file")

for a_tag in a_tags:
    href = 'https://huggingface.co' + a_tag.get('href')
    os.system(f'wget -P /home/{user_name}/{parser.output_dir} {href}')
    print(href)

for obj in json_page:
    href = 'https://huggingface.co/decapoda-research' + obj['security']['repositoryId'][
                                                        obj['security']['repositoryId'].rfind(
                                                            '/'):] + '/resolve/main/' + obj['path']
    # os.system(f'wget {href}')
    os.system(f'wget -P /home/{user_name}/{parser.output_dir} {href}')
    print(href)
