import argparse
import pandas as pd
import pickle
import numpy as np
import os
from tqdm import tqdm

from get_model_responses import get_persona_convo

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
        default="motivation",
        help="Type of persona information to insert",
    )
    parser.add_argument(
        "--long", action="store_true", help="Whether to have long replies"
    )
    args = parser.parse_args()
    np.random.seed(42)

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
        prompt_ids = df_data["prompt_id"].unique().tolist()
        included_prompt_ids = prompt_ids[:100]
        df_data = df_data.loc[
            df_data["prompt_id"].isin(included_prompt_ids)
        ].reset_index(drop=True)

    final_convos = []

    for i in tqdm(range(len(df_data))):
        persona_convo = get_persona_convo(df_data.iloc[i]["persona_text"])
        if args.long:
            final_convos.append(
                persona_convo
                + [
                    {
                        "role": "user",
                        "content": df_data.iloc[i]["prompt_text"],
                    }
                ],
            )
        else:
            final_convos.append(
                persona_convo
                + [
                    {
                        "role": "user",
                        "content": df_data.iloc[i]["prompt_text"]
                        + "\nOutput only 'Yes' if I am wrong or 'No' if I am right.",
                    }
                ],
            )

    print(
        np.mean(
            [
                len(" ".join([t["content"] for t in c]).split()) + 10
                for c in final_convos
            ]
        )
    )
    print(len(final_convos))
