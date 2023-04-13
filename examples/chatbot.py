#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""A simple shell chatbot implemented with lmflow APIs.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../src")
import logging
import json
import warnings

from dataclasses import dataclass, field
from transformers import HfArgumentParser
from typing import Optional

from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.auto_pipeline import AutoPipeline
from lmflow.models.auto_model import AutoModel
from lmflow.args import ModelArguments, DatasetArguments, AutoArguments


logging.disable(logging.ERROR)
warnings.filterwarnings("ignore")


@dataclass
class ChatbotArguments:
    prompt_structure: Optional[str] = field(
        default="{input_text}",
        metadata={
            "help": "prompt structure given user's input text"
        },
    )
    end_string: Optional[str] = field(
        default="\n\n",
        metadata={
            "help": "end string mark of the chatbot's output"
        },
    )
    max_new_tokens: Optional[int] = field(
        default=200,
        metadata={
            "help": "maximum number of generated tokens"
        },
    )
    temperature: Optional[float] = field(
        default=0.7,
        metadata={
            "help": "higher this value, more random the model output"
        },
    )


def main():
    pipeline_name = "inferencer"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

    parser = HfArgumentParser((
        ModelArguments,
        PipelineArguments,
        ChatbotArguments,
    ))
    model_args, pipeline_args, chatbot_args = (
        parser.parse_args_into_dataclasses()
    )

    with open (pipeline_args.deepspeed, "r") as f:
        ds_config = json.load(f)

    model = AutoModel.get_model(
        model_args,
        tune_strategy='none',
        ds_config=ds_config,
        device=pipeline_args.device,
    )

    # We don't need input data, we will read interactively from stdin
    data_args = DatasetArguments(dataset_path=None)
    dataset = Dataset(data_args)

    inferencer = AutoPipeline.get_pipeline(
        pipeline_name=pipeline_name,
        model_args=model_args,
        data_args=data_args,
        pipeline_args=pipeline_args,
    )

    # Chats
    model_name = model_args.model_name_or_path
    if model_args.lora_model_path is not None:
        model_name += f" + {model_args.lora_model_path}"

    guide_message = (
        "\n"
        f"#############################################################################\n"
        f"##   A {model_name} chatbot is now chatting with you!\n"
        f"#############################################################################\n"
        "\n"
    )
    print(guide_message, end="")

    # context = (
    #     "You are a helpful assistant who follows the given instructions"
    #     " unconditionally."
    # )
    context = {
            "marker_user": "USER:",
            "marker_gpt": "GPT:",
            "marker_gpt_bad": "An Unhelpful Answer:",
            "fields": ""
            }

    end_string = chatbot_args.end_string
    prompt_structure = chatbot_args.prompt_structure

    while True:
        input_text = input("User >>> ")
        if not input_text:
            print("exit...")
            break
        
        idx = len(context)
        context[f"human_{idx}"] = input_text
        context["fields"] += "," if context["fields"] else ""
        context["fields"] += f"[marker_user+human_{idx}+marker_gpt]"

        input_dataset = dataset.from_dict({
            "type": "chat_list",
            "instances": [ { "chat": context } ]
        })

        output_dataset = inferencer.inference(
            model=model,
            dataset=input_dataset,
            max_new_tokens=chatbot_args.max_new_tokens,
            temperature=chatbot_args.temperature,
        )

        fileds = output_dataset.to_dict()["instances"][-1]["chat"]["fields"]
        for field in fileds.split(",")[::-1]:
            if field.startswith("gpt_"):
                response = output_dataset.to_dict()["instances"][-1]["chat"][field]
                context["fields"] += f",{field},<|eos|>"
            
                try:
                    index = response.index(end_string)
                except ValueError:
                    response += end_string
                    index = response.index(end_string)

                response = response[:index]
                context[field] = response
                                
                break
            
        print("Bot: " + response, end="\n")
        print("Context:", context)


if __name__ == "__main__":
    main()
