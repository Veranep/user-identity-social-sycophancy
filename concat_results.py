import argparse
import os
import pandas as pd

from evaluate import persona_types

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
        "--long", action="store_true", help="Whether to have long replies"
    )
    args = parser.parse_args()
    df = pd.DataFrame([])
    for persona_type in persona_types:
        if not os.path.isfile(
            f"results/{args.model.split('/')[-1]}_{persona_type}_{'long_' if args.long else ''}results.pkl"
        ):
            continue
        df_pt = pd.read_pickle(
            f"results/{args.model.split('/')[-1]}_{persona_type}_{'long_' if args.long else ''}results.pkl"
        )
        df = pd.concat([df, df_pt], ignore_index=True)
    df.to_pickle(
        f"results/{args.model.split('/')[-1]}_{'long_' if args.long else ''}results.pkl"
    )
