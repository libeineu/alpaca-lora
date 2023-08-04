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


def main(
        load_8bit: bool = False,
        base_model: str = "../alpaca13b",
        lora_weights: str = "/home/v-lbei/alpaca-lora/625_7b",
        prompt_template: str = "vicuna_simplification",  # The prompt template to use, will default to alpaca.
        server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
        share_gradio: bool = False,
        batch_size: int = None,
        test_file: str = "/home/v-lbei/simplification/
        prompt_num: int = 1,
        output_file: str = "./vicuna-simplification-dtg",
        mode: str = "base",
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

        assert mode == "base" or mode == "dtg", "mode must be base or dtg"
        if mode == "base":
            return f"###USER: Please provide the simplification of the following paragraph:\n{src}\n###ASSISTANT:\n{tgt}"
        elif mode == "dtg":
            return f"###USER:\nGiven the English paragraph: {src}\nthe already generated simplification: .\nPlease detect the error type firstly, and provide the refined simplification of the given paragraph\n###ASSISTANT:\nError type: incorrect simplification, the refined simplification is: {tgt}"
        
    def create_dataset(data_store_path, test_data_path,
            src="en", 
            tgt="de",
            prompt_num=1,
            mode='base',
    ):
        test_src = read_lines(f'{test_data_path}.{src}')
        datastore_src = read_lines(f'{data_store_path}.{src}')
        datastore_tgt = read_lines(f'{data_store_path}.{tgt}')

        data_with_prompt = []

        for test_src_line in test_src:
            prompts = []
            for i in range(prompt_num):
                src_line = datastore_src[i]
                tgt_line = datastore_tgt[i]
                prompts.append(format_in_template(src_line, tgt=tgt_line, src_name=src, tgt_name=tgt, mode=mode))
            # prompts.append(format_in_template(test_src_line, src_name=src, tgt_name=tgt))
            data_with_prompt.append("\n".join(prompts))
            
        return data_with_prompt

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
    shot = 0  # "zero" "one" "few"

    # output_file_name = lora_weights.split("/")[-1] + ".out"
    fw_file = open(output_file, "w", encoding="utf-8")

    input_file = open(os.path.join(test_file, "test." + src), "r", encoding="utf-8")

    input_lines = input_file.readlines()
    input_lines_len = len(input_lines)

    batch_list = []
    line_list = []

    dataset = create_dataset(os.path.join(test_file, "dev"), os.path.join(test_file, "test"), src, tgt, prompt_num, mode=mode)

    # print(dataset[0])

    line_index = 0
    shot_line = None
    for line in input_lines:
        line_index += 1
        # instruction = "Translate German to English:"
        # if shot_line is not None:
        #     instruction = instruction + "\n\n" + shot_line
        instruction = dataset[line_index - 1]

        if batch_size is not None and batch_size != 1:
            if line_index % batch_size == 0:
                print("========================")
                print("start generate line {}\n".format(range(line_index - batch_size, line_index - 1)))
                # print("Instruction:", instruction)
                
            line_list.append(line)
            # prompt = prompter.generate_prompt(instruction, "Translate the German sentence into English.\n" + line.strip("\n").strip(" "), mode=mode)
            prompt = prompter.generate_prompt(instruction, line.strip("\n").strip(" "), mode=mode)
            # print(f"prompt is :{prompt}")
            # assert 0
            batch_list.append(prompt)

        else:
            print("========================")
            print(f"start generate line {line_index}\n")
            print("Instruction:", instruction)

        if batch_size is not None and batch_size != 1:
            if line_index % batch_size != 0 and line_index != input_lines_len:
                continue
            else:
                print(f"src list is :{line_list}")
                # print(f"batch list is :{batch_list}")
                # assert 0
                result = evaluate(batch_list, batch_size=batch_size)
                batch_list = []
                for item_id in range(len(result)):
                    result_line = prompter.get_response(result[item_id])
                    # print(f"result_line is :{result_line}")
                    # assert 0
                    print(f"Response--{line_index - batch_size + item_id} : {result_line}")
                    fw_file.write(line_list[item_id].strip("\n").strip(" ") + "\t" + result_line + "\n")
                print("========================")
                line_list = []
        else:
            print(f"src is :{line}")
            result = evaluate(instruction, line.strip("\n").strip(" ") + " => ")
            real_result = result.split("\n")[0].strip("\n").strip(" ")
            print(f"Response: {result}")
            print(f"real_result output is : {real_result}")
            print("========================")
            fw_file.write(line.strip("\n").strip(" ") + "\t" + real_result + "\n")

    input_file.close()
    fw_file.close()
    ###########

    # testing code for readme
    for instruction in [
        "Tell me about alpacas.",
        "Tell me about the president of Mexico in 2019.",
        "Tell me about the king of France in 2019.",
        "List all Canadian provinces in alphabetical order.",
        "Write a Python program that prints the first 10 Fibonacci numbers.",
        "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",
        # noqa: E501
        "Tell me five words that rhyme with 'shock'.",
        "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
        "Count up from 1 to 500.",
    ]:
        print("Instruction:", instruction)
        result = evaluate(instruction, "i have an apple")
        print(f"Response: {result}")
        exit(0)


if __name__ == "__main__":
    fire.Fire(main)
