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
        base_model: str = "",
        lora_weights: str = "tloen/alpaca-lora-7b",
        prompt_template: str = "",  # The prompt template to use, will default to alpaca.
        server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
        share_gradio: bool = False,
):

    print(f"base_model:{base_model}")
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
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

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
            instruction,
            input=None,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=128,
            stream_output=False,
            **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
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
            )

        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return prompter.get_response(output)

    # gr.Interface(
    #     fn=evaluate,
    #     inputs=[
    #         gr.components.Textbox(
    #             lines=2,
    #             label="Instruction",
    #             placeholder="Tell me about alpacas.",
    #         ),
    #         gr.components.Textbox(lines=2, label="Input", placeholder="none"),
    #         gr.components.Slider(
    #             minimum=0, maximum=1, value=0.1, label="Temperature"
    #         ),
    #         gr.components.Slider(
    #             minimum=0, maximum=1, value=0.75, label="Top p"
    #         ),
    #         gr.components.Slider(
    #             minimum=0, maximum=100, step=1, value=40, label="Top k"
    #         ),
    #         gr.components.Slider(
    #             minimum=1, maximum=4, step=1, value=4, label="Beams"
    #         ),
    #         gr.components.Slider(
    #             minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
    #         ),
    #         gr.components.Checkbox(label="Stream output"),
    #     ],
    #     outputs=[
    #         gr.inputs.Textbox(
    #             lines=5,
    #             label="Output",
    #         )
    #     ],
    #     title="🦙🌲 Alpaca-LoRA",
    #     description="Alpaca-LoRA is a 7B-parameter LLaMA model finetuned to follow instructions. It is trained on the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset and makes use of the Huggingface LLaMA implementation. For more information, please visit [the project's website](https://github.com/tloen/alpaca-lora).",  # noqa: E501
    # ).queue().launch(server_name="0.0.0.0", share=share_gradio)
    # # Old testing code follows.


    ###########
    # "translation" "read"
    task_tag = "translation"
    """
    param_translation
    """
    src = "de"
    tgt = "en"
    iwslt_file = "/home/v-lbei/deen"
    shot = 0  # "zero" "one" "few"

    """
    param_read
    """
    # file_dir = "/mnt/shanweiqiao/huggingface_llama/data_llama_test/RACE/"
    file_dir = "/mnt/shanweiqiao/alpaca-lora-main/read_data/RACE_test/"
    data_name = ["test"]
    # data_name = ["train", "dev", "test"]
    domain = ["high", "middle"]
    # domain = ["high", "middle"]
    tag = ["id", "article", "questions", "options", "answers"]
    options_tag = ["A", "B", "C", "D"]


    output_file = open(os.path.join(iwslt_file, "test.alpaca_lora_sys"), "w", encoding="utf-8")


    if task_tag == "read":
        def construct_prompt(real_data, index):
            out_line = "\"\"\"\n"
            out_line += "article: \n"
            out_line = out_line + real_data["article"] + "\n"
            out_line += "questions: \n"
            out_line = out_line + real_data["questions"][index] + "\n"
            out_line += "options: \n"
            for idx in range(len(real_data["options"][index])):
                out_line = out_line + options_tag[idx] + ". " + real_data["options"][index][idx] + "\n"
            out_line += "answers: \"\"\""
            print(f"out_line is : {len(out_line)}")
            return out_line

        for set_item in data_name:
            new_path = os.path.join(file_dir, set_item)
            for domain_item in domain:
                final_path = os.path.join(new_path, domain_item)
                file_list = os.listdir(final_path)
                output_file = open(os.path.join(final_path, "test.alpaca_lora_sys"), "w", encoding="utf-8")
                line_index = 0
                for item in file_list:
                    real_data_path = os.path.join(final_path, item)
                    data_file = open(real_data_path, 'r', encoding="utf-8")
                    real_data = json.load(data_file)
                    for i in range(len(real_data["answers"])):
                        print("========================")
                        line_index += 1
                        print(f"start generate line {line_index}\n")
                        instruction = construct_prompt(real_data, i)
                        print(f"prompt is : \n{instruction}\n")
                        result = evaluate(instruction)
                        real_result = result.split("\n")[0].strip("\n").strip(" ")
                        print(f"Response: {result}")
                        print(f"real_result output is : {real_result}")
                        print("========================")
                    data_file.close()
                output_file.close()

    elif task_tag == "translation":
        input_file = open(os.path.join(iwslt_file, "test." + src), "r", encoding="utf-8")
        train_file_src = open(os.path.join(iwslt_file, "train." + src), "r", encoding="utf-8")
        train_file_tgt = open(os.path.join(iwslt_file, "train." + tgt), "r", encoding="utf-8")

        input_lines = input_file.readlines()

        def random_select_train():
            all_sen_src = train_file_src.readlines()
            all_sen_tgt = train_file_tgt.readlines()
            if shot == 1:
                seed = random.randint(0, len(all_sen_src) - 1)
                src_line = all_sen_src[seed].strip("\n").strip("\t").strip(" ")
                tgt_line = all_sen_tgt[seed].strip("\n").strip("\t").strip(" ")
                shot_line = src_line + " => " + tgt_line + "\n\n"
                print(f"promte few shot is : {shot_line}")
                return shot_line
            elif shot > 1:
                shot_line = ""
                for i in range(shot):
                    seed = random.randint(0, len(all_sen_src) - 1)
                    src_line = all_sen_src[seed].strip("\n").strip("\t").strip(" ")
                    tgt_line = all_sen_tgt[seed].strip("\n").strip("\t").strip(" ")
                    shot_line = shot_line + src_line + " => " + tgt_line + "\n\n"
                print(f"promte few shot is : {shot_line}")
                return shot_line
            train_file_src.close()
            train_file_tgt.close()
            return None

        line_index = 0
        shot_line = random_select_train()
        for line in input_lines:
            line_index += 1
            print("========================")
            print(f"start generate line {line_index}\n")
            instruction = "Translate German to English:"
            if shot_line is not None:
                instruction = instruction + "\n\n" + shot_line
            print("Instruction:", instruction)
            print(f"src is :{line}")
            result = evaluate(instruction, line.strip("\n").strip(" ") + " => ")
            real_result = result.split("\n")[0].strip("\n").strip(" ")
            print(f"Response: {result}")
            print(f"real_result output is : {real_result}")
            print("========================")
            output_file.write(line.strip("\n").strip(" ") + "\t" + real_result + "\n")

        input_file.close()
        train_file_src.close()
        train_file_tgt.close()
        output_file.close()
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
