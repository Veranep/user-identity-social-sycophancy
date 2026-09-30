import argparse
import os
import io
import torch
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

from utils import persona_types, wc_persona_types

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to evaluate",
    )
    # parser.add_argument(
    #     "--long", action="store_true", help="Whether to have long replies"
    # )
    parser.add_argument(
        "--wc",
        action="store_true",
        help="Whether to concatenate wildchat or regular results",
    )
    parser.add_argument(
        "--mitigate",
        action="store_true",
        help="Whether to concatenate mitigation or regular results",
    )
    args = parser.parse_args()
    df = pd.DataFrame([])
    pts = wc_persona_types if args.wc else persona_types
    for persona_type in tqdm(pts):
        # if not os.path.isfile(
        #     f"results/{args.model.split('/')[-1]}_{persona_type}_{'long_' if args.long else ''}{'mitigate_' if args.mitigate else ''}results.pkl"
        # ):
        #     continue
        df_pt = pd.read_pickle(
            f"results/{args.model.split('/')[-1]}_{persona_type}_{'mitigate_' if args.mitigate else ''}results.pkl"
        )
        print(df_pt)
        if "prompt_text" in df_pt.columns:
            df_pt = df_pt.drop(columns=["prompt_text", "persona_text"])
        df_pt.to_pickle(
            f"results/{args.model.split('/')[-1]}_{persona_type}_{'mitigate_' if args.mitigate else ''}results.pkl"
        )
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

    current_df = pd.read_pickle(
        f"results/{args.model.split('/')[-1]}_{'mitigate_' if args.mitigate else ''}{'wc_' if args.wc else ''}results.pkl"
    )

    if not df.equals(current_df):
        print("Saved new df because it's different!")
        df.to_pickle(
            f"results/{args.model.split('/')[-1]}_{'mitigate_' if args.mitigate else ''}{'wc_' if args.wc else ''}results.pkl"
        )
