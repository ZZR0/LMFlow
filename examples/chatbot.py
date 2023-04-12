#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""A simple shell chatbot implemented with lmflow APIs.
"""
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
            "human_1": "Please write a summary of the following paragraph: I am 23 years old, male. My ex-girlfriend is 25. 4 years ago we had a really intense, short relationship, 8 months long. I don't know what love is, but I definitely thought I was in love with her at the time. I feel like we had a very strong emotional connection. However, we both had some serious issues. I was using a lot of drugs. She became very dependent on me for her own well-being and that made me want to distance myself. I broke up with her and it was very hard for her.\n\nI got sober, and a couple of years ago I realized I might be gay. Things made a lot of sense, I thought I was bisexual before but that I could control who I was attracted to. I really didn't want to be with guys so I thought I could just ignore that part of me.\n\nI felt a lot of reluctance to come out because I did have some great sexy times with my ex-GF early in our relationship. But I had a lot of hangups as well. A lot of anxiety surrounding sex.\n\nJust recently I sent my ex-GF an email telling her that I'm pretty sure I'm gay and she told me she was really happy for me and that she thought she had ruined her one chance of true love but really the relationship wouldn't have worked anyways, because I'm gay. That's awesome, and I'm stoked for her that she can have some closure.\n\nAnd so we were planning to meet up and talk. I do really like her and I'm excited. It feels like we are going to be reunited. But now all of a sudden I feel attraction to her, sexual attraction. I want to lay in bed naked with her and have intimate sex.\n\nI feel very confused. I don't want to hurt her but I feel like I want to explore these feelings too. My gut tells me that I shouldn't even tell her about these feelings because I don't want to be in a relationship with her, because I still need to explore the side of me that is attracted to guys. I want to be honest with her as well.",
            "gpt_1": "I was in a relationship with a girl. She became very dependent on me and I broke up with her. I realized I was gay. I feel sexual attraction to my ex-GF and I don't know about the relationship. I feel confused and freaked out.",
            "fields": "[marker_user+human_1+marker_gpt],gpt_1,<|eos|>"
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
                


if __name__ == "__main__":
    main()
