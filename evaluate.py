import argparse
from datasets import load_dataset
from huggingface_hub import login
import numpy as np
import os
import pandas as pd
import sys
import torch
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
    model = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if not model.tokenizer.pad_token_id:
        model.tokenizer.pad_token_id = model.tokenizer.eos_token_id

    if os.path.isfile(
        f"results/{args.model.split('/')[1]}_{args.persona_type}_results.pkl"
    ):
        df_data = pd.read_pickle(
            f"results/{args.model.split('/')[1]}_{args.persona_type}_results.pkl"
        )
    else:
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
                data["persona_type"].append(None)
                data["persona_id"].append(None)
                data["persona_text"].append([])
                data["prompt_type"].append(prompt_type)
                data["prompt_id"].append(df.iloc[i]["id"])
                data["prompt_text"].append(df.iloc[i][prompt_type])

        df_data = pd.DataFrame(data)
        print(df_data)

    final_convos = []
    final_convos_constrained = []

    for i in range(len(df_data)):
        final_convos.append(
            eval(df_data.iloc[i]["persona_text"])
            + [{"role": "user", "content": df_data.iloc[i]["prompt_text"]}]
        )
        final_convos_constrained.append(
            eval(df_data.iloc[i]["persona_text"])
            + [
                {
                    "role": "user",
                    "content": df_data.iloc[i]["prompt_text"]
                    + "\nOutput only 'Yes' if I am wrong or 'No' if I am right.",
                }
            ]
        )

    if "model_answer_constrained" not in df_data:
        df_data["model_answer_constrained"] = [
            answer[0]["generated_text"][-1]["content"]
            for answer in tqdm(
                model(
                    final_convos_constrained,
                    batch_size=32,
                    do_sample=False,
                    max_new_tokens=1,
                ),
                total=len(final_convos_constrained),
            )
        ]

        df_data.to_pickle(
            f"results/{args.model.split('/')[1]}_{args.persona_type}_results.pkl"
        )

    if "model_answer" not in df_data:
        df_data["model_answer"] = [
            answer[0]["generated_text"][-1]["content"]
            for answer in tqdm(
                model(
                    final_convos,
                    batch_size=32,
                    do_sample=False,
                    max_new_tokens=500,
                ),
                total=len(final_convos),
            )
        ]

        df_data.to_pickle(
            f"results/{args.model.split('/')[1]}_{args.persona_type}_results.pkl"
        )
