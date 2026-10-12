import argparse
import os
import io
import torch
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

from utils import persona_types, ic_persona_types

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
        "--ic",
        action="store_true",
        help="Whether to concatenate infinitychat or regular results",
    )
    parser.add_argument(
        "--mitigate",
        action="store_true",
        help="Whether to concatenate mitigation or regular results",
    )
    args = parser.parse_args()
    df = pd.DataFrame([])
    pts = ic_persona_types if args.ic else persona_types
    for persona_type in tqdm(pts):
        df_pt = pd.read_pickle(
            f"results/{args.model.split('/')[-1]}_{persona_type}_{'mitigate_' if args.mitigate else ''}results.pkl"
        )
        print(df_pt)
        df_pt.to_pickle(
            f"results/{args.model.split('/')[-1]}_{persona_type}_{'mitigate_' if args.mitigate else ''}results.pkl"
        )
        # For proprietary models only keep items with two valid answers
        if "model_answer_constrained" in df_pt.columns:
            print(df_pt)
            df_pt = df_pt.loc[
                df_pt["model_answer_constrained"]
                .str.lower()
                .isin(["yes", "no"])
            ]
            print(df_pt)
            df_pt = df_pt[
                df_pt.groupby(
                    ["persona_type", "persona_id", "prompt_id"], dropna=False
                )["prompt_type"]
                .transform("nunique")
                .eq(2)
            ]
            print(df_pt)

        df = pd.concat([df, df_pt], ignore_index=True)

    df.to_pickle(
        f"results/{args.model.split('/')[-1]}_{'mitigate_' if args.mitigate else ''}{'ic_' if args.ic else ''}results.pkl"
    )
