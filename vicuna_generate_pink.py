import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
import sys

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

from torch.utils.data import DataLoader, Dataset
from datasets import Dataset as Dataset2
from collections import defaultdict
import tqdm
from comet import load_from_checkpoint, download_model

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

######
import random
import json
import torch.nn as nn
from typing import List, Optional, Tuple, Union


# from transformers import GenerationMixin
# from transformers.generation import _prepare_attention_mask_for_generation
# from transformers.generation.utils.GenerationMixin import prepare_inputs_for_generation

######


demonstrations = {"step1": "###USER:\nInstruction: Extract the German to English mapping of all phrase pairs from a given translation pair.\nGerman: Im Sommer genieße ich den Duft von frisch geschnittenem Gras.\nEnglish: In summer, I enjoy the scent of freshly cut grass.\n###ASSISTANT:\n\"Im Sommer\" -> \"In summer\", \"genieße ich\" -> \"I enjoy\", \"den Duft\" -> \"the scent\", \"von\" -> \"of\", \"frisch geschnittenem\" -> \"freshly cut\", \"Gras\" -> \"grass\"", "step2": "###USER:\nInstruction: Extract the German to English mapping of all word pairs from a given translation pair.\nGerman: Im Sommer genieße ich den Duft von frisch geschnittenem Gras.\nEnglish: In summer, I enjoy the scent of freshly cut grass.\n###ASSISTANT:\n\"Im\" -> \"In\", \"Sommer\" -> \"summer,\", \"genieße\" -> \"enjoy\", \"ich\" -> \"I\",\"den\" -> \"the\", \"Duft\" -> \"scent\", \"von\" -> \"of\",  \"frisch\" -> \"freshly\", \"geschnittenem\" -> \"cut\", \"Gras\" -> \"grass\""}

comet_model_mapping = {
    "wmt21-comet-qe-da": "models--Unbabel--wmt20-comet-qe-da/snapshots/2e7ffc84fb67d99cf92506611766463bb9230cfb/checkpoints/model.ckpt",
    "wmt22-qe-da": "models--Unbabel--wmt22-comet-da/snapshots/371e9839ca4e213dde891b066cf3080f75ec7e72/checkpoints/model.ckpt",
}

def comet_qe(src_lines, sys_lines, comet_qe_model_name="wmt21-comet-qe-da", batch_size=128, comet_saving_dir="../"):
    data = []
    for sys, src in zip(sys_lines, src_lines):
        data.append({"mt": sys, "src": src, "ref": None})
    if data:
        if comet_qe_model_name in comet_model_mapping:
            comet_model = load_from_checkpoint(os.path.join(comet_saving_dir, comet_model_mapping[comet_qe_model_name]))
        else:
            model_path = download_model(comet_qe_model_name, saving_directory=comet_saving_dir)
            comet_model = load_from_checkpoint(model_path)

        comet_model.eval()
        model_output = comet_model.predict(data, batch_size=batch_size, gpus=1)
        scores = model_output.scores

    return scores

def comet(src_lines, sys_lines, ref_lines, comet_qe_model_name="wmt22-qe-da", batch_size=128, comet_saving_dir="../"):
    data = []
    for sys, src, ref in zip(sys_lines, src_lines, ref_lines):
        data.append({"mt": sys, "src": src, "ref": ref})
    if data:
        if comet_qe_model_name in comet_model_mapping:
            comet_model = load_from_checkpoint(os.path.join(comet_saving_dir, comet_model_mapping[comet_qe_model_name]))
        else:
            model_path = download_model(comet_qe_model_name, saving_directory=comet_saving_dir)
            comet_model = load_from_checkpoint(model_path)

        comet_model.eval()
        model_output = comet_model.predict(data, batch_size=batch_size, gpus=1)
        scores = model_output.scores
    return scores



def main(
        load_8bit: bool = False,
        base_model: str = "circulus/alpaca-7b",
        lora_weights: str = "tloen/alpaca-lora-7b/",
        prompt_template: str = "alpaca",  # The prompt template to use, will default to alpaca.
        server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
        share_gradio: bool = False,
        batch_size: int = None,
        test_file: str = "deen/test",
        prompt_num: int = 1,
        output_file: str = "./output/vicuna7b-v1.5.dtg-5shot-step",
        mode: str = "dtg",
):

    print(f"base_model:{base_model}")
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    # 后面多了一个/r
    lora_weights = lora_weights.strip()
    base_model = base_model.strip()

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model, padding_side="left")
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir="../cache" + base_model,
        )

        # model = PeftModel.from_pretrained(
        #     model,
        #     lora_weights,
        #     torch_dtype=torch.float16,
        # )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    ####
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # padding_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    # model.config.pad_token_id = padding_id
    ####


    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)


    def read_lines(file_):
        with open(file_, 'r') as f:
            lines = f.readlines()
        return [line.strip() for line in lines]

    def format_in_template(src, tgt=None, 
        src_name="en",
        tgt_name="zh",
        mode='base'):
        # line-break includes space, \n, special symbols. use space for now
        if src_name == "en" and tgt_name == "zh":
            src_str = "English"
            tgt_str = "Chinese"
        elif src_name == "zh" and tgt_name == "en":
            src_str = "Chinese"
            tgt_str = "English"
        elif src_name == "de" and tgt_name == "en":
            src_str = "German"
            tgt_str = "English"
        elif src_name == "en" and tgt_name == "de":
            src_str = "English"
            tgt_str = "German"
        elif src_name == "ja" and tgt_name == "en":
            src_str = "Japanese"
            tgt_str = "English"
        elif src_name == "en" and tgt_name == "ja":
            src_str = "English"
            tgt_str = "Japanese"
        elif src_name == "cs" and tgt_name == "en":
            src_str = "Czech"
            tgt_str = "English"
        elif src_name == "en" and tgt_name == "cs":
            src_str = "English"
            tgt_str = "Czech"
        elif src_name == "ru" and tgt_name == "en":
            src_str = "Russian"
            tgt_str = "English"
        elif src_name == "en" and tgt_name == "ru":
            src_str = "English"
            tgt_str = "Russian"
        elif src_name == "en" and tgt_name == "ha":
            src_str = "English"
            tgt_str = "Hausa"
        elif src_name == "ha" and tgt_name == "en":
            src_str = "Hausa"
            tgt_str = "English"
        elif src_name == "de" and tgt_name == "fr":
            src_str = "German"
            tgt_str = "French"
        elif src_name == "fr" and tgt_name == "de":
            src_str = "French"
            tgt_str = "German"
        elif src_name == "is" and tgt_name == "en":
            src_str = "Icelandic "
            tgt_str = "English"
        elif src_name == "en" and tgt_name == "is":
            src_str = "English "
            tgt_str = "Icelandic"
        elif src_name == "uk" and tgt_name == "en":
            src_str = "Ukrainian "
            tgt_str = "English"
        elif src_name == "en" and tgt_name == "uk":
            src_str = "English "
            tgt_str = "Ukrainian"

        else:
            raise NotImplementedError


        # if tgt is not None:
        #     return f"Translate {src_str} to {tgt_str}:\n\n### Input:\n{src} =>\n\n### Response:\n{tgt}"
        # else:
        #     return f"Translate {src_str} to {tgt_str}:\n\n{src} =>\n\n"
        sys_line = ""

        if tgt is not None:
            return f"###USER:\nInstruction: Translate the following {src_str} sentence into {tgt_str}.\n{src}\n###ASSISTANT:\n{tgt}"
        else:
            return f"###USER:\nInstruction: Translate the following {src_str} sentence into {tgt_str}.\n{src}\n###ASSISTANT:\n"

    def format_in_template_extract_word_level_mapping(src, tgt=None, init_translation=None,
                           src_name="en",
                           tgt_name="zh",
                           mode='base'):
        # line-break includes space, \n, special symbols. use space for now
        if src_name == "en" and tgt_name == "zh":
            src_str = "English"
            tgt_str = "Chinese"
        elif src_name == "zh" and tgt_name == "en":
            src_str = "Chinese"
            tgt_str = "English"
        elif src_name == "de" and tgt_name == "en":
            src_str = "German"
            tgt_str = "English"
        elif src_name == "en" and tgt_name == "de":
            src_str = "English"
            tgt_str = "German"
        elif src_name == "ja" and tgt_name == "en":
            src_str = "Japanese"
            tgt_str = "English"
        elif src_name == "en" and tgt_name == "ja":
            src_str = "English"
            tgt_str = "Japanese"
        elif src_name == "cs" and tgt_name == "en":
            src_str = "Czech"
            tgt_str = "English"
        elif src_name == "en" and tgt_name == "cs":
            src_str = "English"
            tgt_str = "Czech"
        elif src_name == "ru" and tgt_name == "en":
            src_str = "Russian"
            tgt_str = "English"
        elif src_name == "en" and tgt_name == "ru":
            src_str = "English"
            tgt_str = "Russian"
        elif src_name == "en" and tgt_name == "ha":
            src_str = "English"
            tgt_str = "Hausa"
        elif src_name == "ha" and tgt_name == "en":
            src_str = "Hausa"
            tgt_str = "English"
        elif src_name == "de" and tgt_name == "fr":
            src_str = "German"
            tgt_str = "French"
        elif src_name == "fr" and tgt_name == "de":
            src_str = "French"
            tgt_str = "German"
        elif src_name == "is" and tgt_name == "en":
            src_str = "Icelandic "
            tgt_str = "English"
        elif src_name == "en" and tgt_name == "is":
            src_str = "English "
            tgt_str = "Icelandic"
        elif src_name == "uk" and tgt_name == "en":
            src_str = "Ukrainian "
            tgt_str = "English"
        elif src_name == "en" and tgt_name == "uk":
            src_str = "English "
            tgt_str = "Ukrainian"

        else:
            raise NotImplementedError

        # if tgt is not None:
        #     return f"Translate {src_str} to {tgt_str}:\n\n### Input:\n{src} =>\n\n### Response:\n{tgt}"
        # else:
        #     return f"Translate {src_str} to {tgt_str}:\n\n{src} =>\n\n"
        sys_line = ""

        if tgt is not None:
            return f"###USER:\nInstruction: Extract the {src_str} to {tgt_str} mapping of all word pairs from a given translation pair.\n{src_str}: {src}\n{tgt_str}: {init_translation}\n###ASSISTANT:\n {tgt}"
        else:
            return f"###USER:\nInstruction: Extract the {src_str} to {tgt_str} mapping of all word pairs from a given translation pair.\n{src_str}: {src}\n{tgt_str}: {init_translation}\n###ASSISTANT:\n "



    def format_in_template_extract_phrase_level_mapping(src, tgt=None, init_translation=None,
                           src_name="en",
                           tgt_name="zh",
                           mode='base'):
        # line-break includes space, \n, special symbols. use space for now
        if src_name == "en" and tgt_name == "zh":
            src_str = "English"
            tgt_str = "Chinese"
        elif src_name == "zh" and tgt_name == "en":
            src_str = "Chinese"
            tgt_str = "English"
        elif src_name == "de" and tgt_name == "en":
            src_str = "German"
            tgt_str = "English"
        elif src_name == "en" and tgt_name == "de":
            src_str = "English"
            tgt_str = "German"
        elif src_name == "ja" and tgt_name == "en":
            src_str = "Japanese"
            tgt_str = "English"
        elif src_name == "en" and tgt_name == "ja":
            src_str = "English"
            tgt_str = "Japanese"
        elif src_name == "cs" and tgt_name == "en":
            src_str = "Czech"
            tgt_str = "English"
        elif src_name == "en" and tgt_name == "cs":
            src_str = "English"
            tgt_str = "Czech"
        elif src_name == "ru" and tgt_name == "en":
            src_str = "Russian"
            tgt_str = "English"
        elif src_name == "en" and tgt_name == "ru":
            src_str = "English"
            tgt_str = "Russian"
        elif src_name == "en" and tgt_name == "ha":
            src_str = "English"
            tgt_str = "Hausa"
        elif src_name == "ha" and tgt_name == "en":
            src_str = "Hausa"
            tgt_str = "English"
        elif src_name == "de" and tgt_name == "fr":
            src_str = "German"
            tgt_str = "French"
        elif src_name == "fr" and tgt_name == "de":
            src_str = "French"
            tgt_str = "German"
        elif src_name == "is" and tgt_name == "en":
            src_str = "Icelandic "
            tgt_str = "English"
        elif src_name == "en" and tgt_name == "is":
            src_str = "English "
            tgt_str = "Icelandic"
        elif src_name == "uk" and tgt_name == "en":
            src_str = "Ukrainian "
            tgt_str = "English"
        elif src_name == "en" and tgt_name == "uk":
            src_str = "English "
            tgt_str = "Ukrainian"

        else:
            raise NotImplementedError

        # if tgt is not None:
        #     return f"Translate {src_str} to {tgt_str}:\n\n### Input:\n{src} =>\n\n### Response:\n{tgt}"
        # else:
        #     return f"Translate {src_str} to {tgt_str}:\n\n{src} =>\n\n"
        sys_line = ""

        if tgt is not None:
            return f"###USER:\nInstruction: Extract the {src_str} to {tgt_str} mapping of all phrase pairs from a given translation pair.\n{src_str}: {src}\n{tgt_str}: {init_translation}\n###ASSISTANT:\n {tgt}"
        else:
            return f"###USER:\nInstruction: Extract the {src_str} to {tgt_str} mapping of all phrase pairs from a given translation pair.\n{src_str}: {src}\n{tgt_str}: {init_translation}\n###ASSISTANT:\n "

    def format_in_template_translation(src, tgt=None, phrase_level_mappings=None, word_level_mappings=None, sentence_level_mappings=None,
                           src_name="en",
                           tgt_name="zh",
                           mode='base'):
        # line-break includes space, \n, special symbols. use space for now
        if src_name == "en" and tgt_name == "zh":
            src_str = "English"
            tgt_str = "Chinese"
        elif src_name == "zh" and tgt_name == "en":
            src_str = "Chinese"
            tgt_str = "English"
        elif src_name == "de" and tgt_name == "en":
            src_str = "German"
            tgt_str = "English"
        elif src_name == "en" and tgt_name == "de":
            src_str = "English"
            tgt_str = "German"
        elif src_name == "ja" and tgt_name == "en":
            src_str = "Japanese"
            tgt_str = "English"
        elif src_name == "en" and tgt_name == "ja":
            src_str = "English"
            tgt_str = "Japanese"
        elif src_name == "cs" and tgt_name == "en":
            src_str = "Czech"
            tgt_str = "English"
        elif src_name == "en" and tgt_name == "cs":
            src_str = "English"
            tgt_str = "Czech"
        elif src_name == "ru" and tgt_name == "en":
            src_str = "Russian"
            tgt_str = "English"
        elif src_name == "en" and tgt_name == "ru":
            src_str = "English"
            tgt_str = "Russian"
        elif src_name == "en" and tgt_name == "ha":
            src_str = "English"
            tgt_str = "Hausa"
        elif src_name == "ha" and tgt_name == "en":
            src_str = "Hausa"
            tgt_str = "English"
        elif src_name == "de" and tgt_name == "fr":
            src_str = "German"
            tgt_str = "French"
        elif src_name == "fr" and tgt_name == "de":
            src_str = "French"
            tgt_str = "German"
        elif src_name == "is" and tgt_name == "en":
            src_str = "Icelandic "
            tgt_str = "English"
        elif src_name == "en" and tgt_name == "is":
            src_str = "English "
            tgt_str = "Icelandic"
        elif src_name == "uk" and tgt_name == "en":
            src_str = "Ukrainian "
            tgt_str = "English"
        elif src_name == "en" and tgt_name == "uk":
            src_str = "English "
            tgt_str = "Ukrainian"

        else:
            raise NotImplementedError

        # if tgt is not None:
        #     return f"Translate {src_str} to {tgt_str}:\n\n### Input:\n{src} =>\n\n### Response:\n{tgt}"
        # else:
        #     return f"Translate {src_str} to {tgt_str}:\n\n{src} =>\n\n"
        sentence_level_mappings_str = src + " - " + sentence_level_mappings
        sys_line = ""

        if tgt is not None:
            return f"###USER:\nKey Sentence-level mappings: {sentence_level_mappings_str}\nKey Phrase-level mappings: {phrase_level_mappings}\nKey Word-level mappings: {word_level_mappings}\nInstruction: Combine the above knowledge to produce a high-quality {tgt_str} translation of the following {src_str} text.\n{src_str}: {src}\n###ASSISTANT:\n{tgt_str}: {tgt}"
        else:
            return f"###USER:\nKey Sentence-level mappings: {sentence_level_mappings_str}\nKey Phrase-level mappings: {phrase_level_mappings}\nKey Word-level mappings: {word_level_mappings}\nInstruction: Combine the above knowledge to produce a high-quality {tgt_str} translation of the following {src_str} text.\n{src_str}: {src}\n###ASSISTANT:\n{tgt_str}: "

    def format_in_template_translation_two_fold_knowledge(src, tgt=None, phrase_level_mappings=None, word_level_mappings=None,
                                                      sentence_level_mappings=None, phrase_level_mappings_LLM=None,
                                                      word_level_mappings_LLM=None, sentence_level_mappings_LLM=None,
                                                      score_llm=None, score_nmt=None,
                           src_name="en",
                           tgt_name="zh",
                           mode='base'):
        # line-break includes space, \n, special symbols. use space for now
        if src_name == "en" and tgt_name == "zh":
            src_str = "English"
            tgt_str = "Chinese"
        elif src_name == "zh" and tgt_name == "en":
            src_str = "Chinese"
            tgt_str = "English"
        elif src_name == "de" and tgt_name == "en":
            src_str = "German"
            tgt_str = "English"
        elif src_name == "en" and tgt_name == "de":
            src_str = "English"
            tgt_str = "German"
        elif src_name == "ja" and tgt_name == "en":
            src_str = "Japanese"
            tgt_str = "English"
        elif src_name == "en" and tgt_name == "ja":
            src_str = "English"
            tgt_str = "Japanese"
        elif src_name == "cs" and tgt_name == "en":
            src_str = "Czech"
            tgt_str = "English"
        elif src_name == "en" and tgt_name == "cs":
            src_str = "English"
            tgt_str = "Czech"
        elif src_name == "ru" and tgt_name == "en":
            src_str = "Russian"
            tgt_str = "English"
        elif src_name == "en" and tgt_name == "ru":
            src_str = "English"
            tgt_str = "Russian"
        elif src_name == "en" and tgt_name == "ha":
            src_str = "English"
            tgt_str = "Hausa"
        elif src_name == "ha" and tgt_name == "en":
            src_str = "Hausa"
            tgt_str = "English"
        elif src_name == "de" and tgt_name == "fr":
            src_str = "German"
            tgt_str = "French"
        elif src_name == "fr" and tgt_name == "de":
            src_str = "French"
            tgt_str = "German"
        elif src_name == "is" and tgt_name == "en":
            src_str = "Icelandic "
            tgt_str = "English"
        elif src_name == "en" and tgt_name == "is":
            src_str = "English "
            tgt_str = "Icelandic"
        elif src_name == "uk" and tgt_name == "en":
            src_str = "Ukrainian "
            tgt_str = "English"
        elif src_name == "en" and tgt_name == "uk":
            src_str = "English "
            tgt_str = "Ukrainian"

        else:
            raise NotImplementedError

        # if tgt is not None:
        #     return f"Translate {src_str} to {tgt_str}:\n\n### Input:\n{src} =>\n\n### Response:\n{tgt}"
        # else:
        #     return f"Translate {src_str} to {tgt_str}:\n\n{src} =>\n\n"

        sentence_level_mappings_str = src + " - " + sentence_level_mappings
        sentence_level_mappings_LLM_str = src + " - " + sentence_level_mappings_LLM
        # word_level_mappings_str = "1. " + word_level_mappings[0] + "2. " + word_level_mappings[1] + '3. ' + word_level_mappings[2]

        utilize_llm = math.exp(score_llm) / (math.exp(score_llm) + math.exp(score_nmt))
        utilize_nmt = math.exp(score_nmt) / (math.exp(score_llm) + math.exp(score_nmt))
        sys_line = ""

        if tgt is not None:
            return f"###USER:\nNMT Knowledge\nKey Sentence-level mappings: {sentence_level_mappings_str}\nKey Phrase-level mappings: {phrase_level_mappings}\nKey Word-level mappings: {word_level_mappings}\n\n\nLLM Knowledge: Key Sentence-level mappings: {sentence_level_mappings_LLM_str}\nKey Phrase-level mappings: {phrase_level_mappings_LLM}\nKey Word-level mappings: {word_level_mappings_LLM}\n\n\nUtilization Rule: utilize {utilize_llm} LLM knowledge and {utilize_nmt} NMT knowledge.\nInstruction: Combine the above knowledge with above utilization rule to produce a high-quality {tgt_str} translation of the following {src_str} text.\n{src_str}: {src}\n###ASSISTANT:\n{tgt_str}: {tgt}"
        else:
            return f"###USER:\nNMT Knowledge\nKey Sentence-level mappings: {sentence_level_mappings_str}\nKey Phrase-level mappings: {phrase_level_mappings}\nKey Word-level mappings: {word_level_mappings}\n\n\nLLM Knowledge: Key Sentence-level mappings: {sentence_level_mappings_LLM_str}\nKey Phrase-level mappings: {phrase_level_mappings_LLM}\nKey Word-level mappings: {word_level_mappings_LLM}\n\n\nUtilization Rule: utilize {utilize_llm} LLM knowledge and {utilize_nmt} NMT knowledge.\nInstruction: Combine the above knowledge with above utilization rule to produce a high-quality {tgt_str} translation of the following {src_str} text.\n{src_str}: {src}\n###ASSISTANT:\n{tgt_str}: "


    def create_dataset(data_store_path, test_data_path,
            src="en", 
            tgt="de",
            prompt_num=1,
            mode='base',
    ):

        test_src = read_lines(f'{test_data_path}.{src}')  # source
        test_tgt_NMT = read_lines(f'{test_data_path}.{tgt}.WMT22_BEST')  # gpt-3.5 tempire 0.3

        #  generate style
        data_src = []
        data_tgt_NMT = []

        for test_src_line, test_tgt_NMT_line in zip(
                test_src, test_tgt_NMT):
            data_src.append(test_src_line)
            data_tgt_NMT.append(test_tgt_NMT_line)
        return data_src, data_tgt_NMT

    def evaluate(
            instruction,
            input=None,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=128,
            stream_output=False,
            batch_size=None,
            my_pad_id=None,
            **kwargs,
    ):
        if batch_size is not None:
            inputs = tokenizer(instruction, return_tensors="pt", padding=True)
        else:
            prompt = prompter.generate_prompt(instruction, input)
            inputs = tokenizer(prompt, return_tensors="pt")

        input_ids = inputs["input_ids"].to(device)
        inputs_mask = inputs["attention_mask"].to(device)

        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }
        if stream_output:
            # Stream the reply 1 token at a time.
            # This is based on the trick of using 'stopping_criteria' to create an iterator,
            # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

            def generate_with_callback(callback=None, **kwargs):
                kwargs.setdefault(
                    "stopping_criteria", transformers.StoppingCriteriaList()
                )
                kwargs["stopping_criteria"].append(
                    Stream(callback_func=callback)
                )
                with torch.no_grad():
                    model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(
                    generate_with_callback, kwargs, callback=None
                )

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    # new_tokens = len(output) - len(input_ids[0])
                    decoded_output = tokenizer.decode(output)

                    if output[-1] in [tokenizer.eos_token_id]:
                        break

                    return prompter.get_response(decoded_output)
            return  # early return for stream_output

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                attention_mask=inputs_mask,
            )

        if batch_size is not None:
            s = generation_output.sequences
            output = tokenizer.batch_decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            return output
        else:
            s = generation_output.sequences[0]
            output = tokenizer.decode(s)
            return prompter.get_response(output)

    task_tag = "translation"
    src = "de"
    tgt = "en"
    shot = 1  # "zero" "one" "few"

    # output_file_name = lora_weights.split("/")[-1] + ".out"
    fw_file = open(output_file, "w", encoding="utf-8")

    fout1 = open(output_file + ".NMT_words", 'w')
    fout2 = open(output_file + ".NMT_phrases", 'w')
    fout3 = open(output_file + ".LLM_init", 'w')
    fout4 = open(output_file + ".LLM_words", 'w')
    fout5 = open(output_file + ".LLM_phrases", 'w')
    fout6 = open(output_file + ".final_trans", 'w')

    input_file = open(os.path.join(test_file, "test." + src), "r", encoding="utf-8")

    input_lines = input_file.readlines()
    input_lines_len = len(input_lines)

    batch_list = []
    line_list = []

    data_src, data_tgt_NMT = create_dataset(os.path.join(test_file, "dev"), os.path.join(test_file, "test"), src, tgt, prompt_num, mode=mode)

    # print(dataset[0])
    # assert 0

    line_index = 0
    shot_line = None

    data_tgt_LLM = []
    data_src_tgt_phrases_LLM = []
    data_src_tgt_words_LLM = []
    data_src_tgt_phrases_NMT = []
    data_src_tgt_words_NMT = []

    for line in tqdm.tqdm(input_lines):
        line_index += 1
        # instruction = "Translate German to English:"
        # if shot_line is not None:
        #     instruction = instruction + "\n\n" + shot_line

        print("========================")
        print(f"start generate line {line_index}\n")
        # print("Instruction:", instruction)
        print(f"src is :{line}")
        instruction_extract_word_mappings_nmt = demonstrations["step2"] + "\n"+format_in_template_extract_word_level_mapping(src_name=src, tgt_name=tgt, src=data_src[line_index-1], init_translation=data_tgt_NMT[line_index-1])
        print(instruction_extract_word_mappings_nmt)
        result_word_mappings_nmt = evaluate(instruction_extract_word_mappings_nmt, line.strip("\n").strip(" ") + " => ")
        # print(result_word_mappings_nmt)
        real_result_word_mappings_nmt = result_word_mappings_nmt.split("\n")[0].strip("\n").strip(" ")
        # print(real_result_word_mappings_nmt)
        # print(f"Response: {result_word_mappings_nmt}")
        fout1.write(real_result_word_mappings_nmt + '\n')
        print(f"real_result output is : {real_result_word_mappings_nmt}")
        data_src_tgt_words_NMT.append(real_result_word_mappings_nmt)

        instruction_extract_phrase_mappings_nmt = demonstrations["step1"] + "\n"+format_in_template_extract_phrase_level_mapping(src_name=src,
                                                                                              tgt_name=tgt,
                                                                                              src=data_src[
                                                                                                  line_index - 1],
                                                                                              init_translation=
                                                                                              data_tgt_NMT[
                                                                                                  line_index - 1])
        print(instruction_extract_phrase_mappings_nmt)
        result_phrase_mappings_nmt = evaluate(instruction_extract_phrase_mappings_nmt, line.strip("\n").strip(" ") + " => ")
        real_result_phrase_mappings_nmt = result_phrase_mappings_nmt.split("\n")[0].strip("\n").strip(" ")
        fout2.write(real_result_phrase_mappings_nmt + '\n')
        # print(f"Response: {result_phrase_mappings_nmt}")
        print(f"real_result output is : {real_result_phrase_mappings_nmt}")
        data_src_tgt_phrases_NMT.append(real_result_phrase_mappings_nmt)

        instruction_init_LLM = format_in_template(src_name=src, tgt_name=tgt, src=data_src[line_index - 1])
        print(instruction_init_LLM)
        result_init_LLM = evaluate(instruction_init_LLM,
                                              line.strip("\n").strip(" ") + " => ")
        real_result_init_LLM = result_init_LLM.split("\n")[0].strip("\n").strip(" ")
        fout3.write(real_result_init_LLM + '\n')
        # print(f"Response: {result_init_LLM}")
        print(f"real_result output is : {real_result_init_LLM}")
        data_tgt_LLM.append(real_result_init_LLM)

        instruction_extract_word_mappings_LLM = demonstrations["step2"] + "\n"+ format_in_template_extract_word_level_mapping(src_name=src,
                                                                                              tgt_name=tgt,
                                                                                              src=data_src[
                                                                                                  line_index - 1],
                                                                                              init_translation=real_result_init_LLM)
        print(instruction_extract_word_mappings_LLM)
        result_word_mappings_llm = evaluate(instruction_extract_word_mappings_LLM, line.strip("\n").strip(" ") + " => ")
        real_result_word_mappings_llm = result_word_mappings_llm.split("\n")[0].strip("\n").strip(" ")
        # print(f"Response: {result_word_mappings_llm}")
        print(f"real_result output is : {real_result_word_mappings_llm}")
        fout4.write(real_result_word_mappings_llm + '\n')
        data_src_tgt_words_LLM.append(real_result_word_mappings_llm)


        instruction_extract_phrase_mappings_llm = demonstrations["step1"] + "\n"+ format_in_template_extract_phrase_level_mapping(src_name=src,
                                                                                                tgt_name=tgt,
                                                                                                src=data_src[
                                                                                                    line_index - 1],
                                                                                                init_translation=real_result_init_LLM)
        print(instruction_extract_phrase_mappings_llm)
        result_phrase_mappings_llm = evaluate(instruction_extract_phrase_mappings_llm,
                                              line.strip("\n").strip(" ") + " => ")
        real_result_phrase_mappings_llm = result_phrase_mappings_llm.split("\n")[0].strip("\n").strip(" ")

        fout5.write(real_result_phrase_mappings_llm + '\n')
        # print(f"Response: {result_phrase_mappings_llm}")
        print(f"real_result output is : {real_result_phrase_mappings_llm}")
        data_src_tgt_phrases_LLM.append(real_result_phrase_mappings_llm)



        


        # print(f"src is :{line}")
        # result = evaluate(instruction, line.strip("\n").strip(" ") + " => ")
        # real_result = result.split("\n")[0].strip("\n").strip(" ")
        # print(f"Response: {result}")
        # print(f"real_result output is : {real_result}")
        # print("========================")
        # fw_file.write(line.strip("\n").strip(" ") + "\t" + real_result + "\n")

    data_comet_NMT = comet_qe(src_lines=data_src, sys_lines=data_tgt_NMT)
    data_comet_LLM = comet_qe(src_lines=data_src, sys_lines=data_tgt_LLM)

    for input_src, input_tgt_NMT, input_tgt_LLM, input_phrase_NMT, input_word_NMT, input_phrase_LLM, input_word_LLM, input_comet_nmt, input_comet_llm in tqdm.tqdm(
            zip(data_src, data_tgt_NMT, data_tgt_LLM, data_src_tgt_phrases_NMT, data_src_tgt_words_NMT,
                data_src_tgt_phrases_LLM, data_src_tgt_words_LLM, data_comet_NMT, data_comet_LLM)):

        score_NMT = input_comet_nmt
        score_LLM = input_comet_llm

        if score_NMT > score_LLM:
            instruction_translation = format_in_template_translation(src=input_src,
                                                            phrase_level_mappings=input_phrase_NMT,
                                                            word_level_mappings=input_word_NMT,
                                                            sentence_level_mappings=input_tgt_NMT,
                                                            src_name=src, tgt_name=tgt)

            result_translation = evaluate(instruction_translation,
                                                  line.strip("\n").strip(" ") + " => ")
            real_result_translation = result_translation.split("\n")[0].strip("\n").strip(" ")
            # print(f"Response: {result_translation}")
            # print(f"real_result output is : {real_result_translation}")
            fout6.write(real_result_translation + '\n')

        elif score_NMT < score_LLM:

            instruction_translation =  format_in_template_translation(src=input_src,
                                                            phrase_level_mappings=input_phrase_LLM,
                                                            word_level_mappings=input_word_LLM,
                                                            sentence_level_mappings=input_tgt_LLM,
                                                            src_name=src, tgt_name=tgt)
            result_translation = evaluate(instruction_translation,
                                          line.strip("\n").strip(" ") + " => ")
            real_result_translation = result_translation.split("\n")[0].strip("\n").strip(" ")
            fout6.write(real_result_translation + '\n')
            # print(f"Response: {result_translation}")
            # print(f"real_result output is : {real_result_translation}")
        else:

            instruction_translation = format_in_template_translation_two_fold_knowledge(src=input_src,
                                                                               phrase_level_mappings=input_phrase_NMT,
                                                                               word_level_mappings=input_word_NMT,
                                                                               sentence_level_mappings=input_tgt_NMT,
                                                                               phrase_level_mappings_LLM=input_phrase_LLM,
                                                                               word_level_mappings_LLM=input_word_LLM,
                                                                               sentence_level_mappings_LLM=input_tgt_LLM,
                                                                               score_llm=score_LLM, score_nmt=score_NMT,
                                                                               src_name=src, tgt_name=tgt)
            result_translation = evaluate(instruction_translation,
                                          line.strip("\n").strip(" ") + " => ")
            real_result_translation = result_translation.split("\n")[0].strip("\n").strip(" ")
            fout6.write(real_result_translation + '\n')
            # print(f"Response: {result_translation}")
            # print(f"real_result output is : {real_result_translation}")


    input_file.close()
    fout1.close()
    fout2.close()
    fout3.close()
    fout4.close()
    fout5.close()
    fout6.close()

    ###########

    # # testing code for readme
    # for instruction in [
    #     "Tell me about alpacas.",
    #     "Tell me about the president of Mexico in 2019.",
    #     "Tell me about the king of France in 2019.",
    #     "List all Canadian provinces in alphabetical order.",
    #     "Write a Python program that prints the first 10 Fibonacci numbers.",
    #     "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",
    #     # noqa: E501
    #     "Tell me five words that rhyme with 'shock'.",
    #     "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
    #     "Count up from 1 to 500.",
    # ]:
    #     print("Instruction:", instruction)
    #     result = evaluate(instruction, "i have an apple")
    #     print(f"Response: {result}")
    #     exit(0)


if __name__ == "__main__":
    fire.Fire(main)
