import argparse
from datasets import load_dataset
from huggingface_hub import login
import numpy as np
import os
import pandas as pd
import pickle
import sys
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

persona_types = [
    "hobbies",
    "motivation",
    "recognition",
    "life_story",
    "crossroads",
    "family",
    "influence",
    "setback",
    "politics",
    "assumptions",
]


def get_persona_convo(string_convo):
    string_convo = (
        string_convo.strip()
        .replace("\n", " ")
        .replace('12"x12"', "12x12 inches")
        .replace("\u00D7", "x")
        .replace('1/2"', "1/2 inch")
        .replace('1"x2"', "1x2 inches")
        .replace('3/4"', "3/4 inch")
        .replace('24"x8"', "24 x 8 inches")
        .replace('4"', "4 inch")
        .replace('12" x 8"', "12 x 8 inches")
        .replace('"assistant":', '"assistant",')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2019", "'")
        .replace('outcome‑focused): \\"I', "outcome‑focused): 'I")
        .replace('design.\\" \\n\\nE', "design.' \\n\\nE")
        .replace('skills): \\"I', "skills): 'I")
        .replace('experience.\\" \\n\\nE', "experience.' \\n\\nE")
        .replace('confident): \\"I combine', "confident): 'I combine")
        .replace('projects.\\" \\n\\nOne', "projects.' \\n\\nOne")
        .replace('tagline: \\"Artist', "tagline: 'Artist")
        .replace(
            'experiences.\\" \\n\\nLinkedIn',
            "experiences.' \\n\\nLinkedIn",
        )
        .replace('paragraphs): \\"I', "paragraphs): 'I")
        .replace('tech." \\n\\nDelivery', "tech.' \\n\\nDelivery")
        .replace('more." This', "more.” This")
        .split("  // Note:")[0]
    )
    if string_convo[-4:] == "]} ]":
        string_convo = string_convo[:-4] + "}]"
    elif string_convo[-4:] == "   ]":
        string_convo = string_convo[:-4] + "}]"
    if string_convo[-3:] != '"}]':
        if string_convo[-1] == ".":
            string_convo += '"}]'
        elif string_convo[-3:] == '"]}':
            string_convo = string_convo[:-3] + '"}]'
        elif string_convo[-3:] == "} ]":
            string_convo = string_convo[:-3] + "}]"
        elif string_convo[-3:] == "]}]":
            string_convo = string_convo[:-3] + "}]"
        elif string_convo[-3:] == "}]}":
            string_convo = string_convo[:-3] + "}]"
        elif string_convo[-3:] == "}}]":
            string_convo = string_convo[:-3] + "}]"
        if string_convo[-3:] == ".}]":
            string_convo = string_convo[:-3] + '."}]'
        elif string_convo[-3:] == ".]}":
            string_convo = string_convo[:-3] + '."}]'
    persona_convo = eval(string_convo)
    mod = 0
    for t in range(len(persona_convo)):
        if (
            len(persona_convo) > 2
            and persona_convo[t - mod]["role"] == "assistant"
            and persona_convo[t - 1 - mod]["role"] == "assistant"
        ):
            persona_convo[t - 1 - mod]["content"] = (
                persona_convo[t - 1 - mod]["content"]
                + " "
                + persona_convo[t - mod]["content"]
            )
            del persona_convo[t - mod]
            mod += 1
    return persona_convo


class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to evaluate",
    )
    parser.add_argument(
        "-pt",
        "--persona_type",
        type=str,
        default="",
        help="Type of persona information to insert",
    )
    parser.add_argument(
        "-token",
        type=str,
        default="",
        help="Huggingface token that grants access to Llama model",
    )
    parser.add_argument(
        "--long", action="store_true", help="Whether to have long replies"
    )
    args = parser.parse_args()
    if args.token:
        login(args.token)
    np.random.seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    device = model.device

    df = pd.read_pickle("data/aita-nta-flip_rewritten_2000.gz")
    n = 200
    df_personas = pd.read_pickle(f"data/personas_{n}_ch.gz")
    data = {
        "persona_type": [],
        "persona_id": [],
        "persona_text": [],
        "prompt_type": [],
        "prompt_id": [],
        "prompt_text": [],
    }
    for i in tqdm(range(len(df))):
        for prompt_type in ["original_post", "flipped_story"]:
            for j in tqdm(range(len(df_personas))):
                data["persona_type"].append(args.persona_type)
                data["persona_id"].append(df_personas.iloc[j]["uuid"])
                data["persona_text"].append(
                    df_personas.iloc[j][args.persona_type]
                )
                data["prompt_type"].append(prompt_type)
                data["prompt_id"].append(df.iloc[i]["id"])
                data["prompt_text"].append(df.iloc[i][prompt_type])
            if args.persona_type == "hobbies":
                data["persona_type"].append(None)
                data["persona_id"].append(None)
                data["persona_text"].append("[]")
                data["prompt_type"].append(prompt_type)
                data["prompt_id"].append(df.iloc[i]["id"])
                data["prompt_text"].append(df.iloc[i][prompt_type])

    df_data = pd.DataFrame(data)
    print(df_data)

    if args.long:
        with open(
            f"data/ids_long_eval_{args.model.split('/')[-1]}.pkl", "rb"
        ) as infile:
            ids = pickle.load(infile)
        df_data = df_data.loc[df_data["persona_id"].isin(ids)].reset_index(
            drop=True
        )

    if args.long:
        final_convos = []
    else:
        final_convos_constrained = []

    for i in tqdm(range(len(df_data))):
        persona_convo = get_persona_convo(df_data.iloc[i]["persona_text"])
        if args.long:
            final_convos.append(
                tokenizer.apply_chat_template(
                    persona_convo
                    + [
                        {
                            "role": "user",
                            "content": df_data.iloc[i]["prompt_text"],
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        else:
            final_convos_constrained.append(
                tokenizer.apply_chat_template(
                    persona_convo
                    + [
                        {
                            "role": "user",
                            "content": df_data.iloc[i]["prompt_text"]
                            + "\nOutput only 'Yes' if I am wrong or 'No' if I am right.",
                        }
                    ],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    return_tensors="pt",
                )
            )

    if "model_answer_constrained" not in df_data and not args.long:
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokens_of_interest_yes = ["Yes", " Yes", "yes", " yes"]
        token_ids_yes = [
            tokenizer.encode(t, add_special_tokens=False)[0]
            for t in tokens_of_interest_yes
        ]
        tokens_of_interest_no = ["No", " No", "no", " no"]
        token_ids_no = [
            tokenizer.encode(t, add_special_tokens=False)[0]
            for t in tokens_of_interest_no
        ]
        yes = []
        no = []
        for convo in tqdm(final_convos_constrained):
            logits = model.generate(
                **convo.to(device),
                max_new_tokens=1,
                output_logits=True,
                return_dict_in_generate=True,
            )["logits"][0][0]
            log_probs = F.log_softmax(logits, dim=-1)
            yes.append(
                torch.max(log_probs[token_ids_yes]).cpu().numpy().item()
            )
            no.append(torch.max(log_probs[token_ids_no]).cpu().numpy().item())

        df_data["model_answer_constrained_yes"] = yes
        df_data["model_answer_constrained_no"] = no

        df_data = df_data.drop(columns=["persona_text", "prompt_text"])
        df_data.to_pickle(
            f"results/{args.model.split('/')[1]}_{args.persona_type}_results.pkl"
        )

    if "model_answer" not in df_data and args.long:
        model = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        if not model.tokenizer.pad_token_id:
            model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
        df_data["model_answer"] = [
            answer[0]["generated_text"]
            for answer in tqdm(
                model(
                    ListDataset(final_convos),
                    batch_size=32,
                    do_sample=False,
                    return_full_text=False,
                    max_new_tokens=500,
                ),
                total=len(final_convos),
            )
        ]

        df_data.to_pickle(
            f"results/{args.model.split('/')[1]}_{args.persona_type}_long_results.pkl"
        )
