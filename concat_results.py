import argparse
import os
import io
import torch
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

from get_model_responses import persona_types


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        return super().find_class(module, name)


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
    for persona_type in tqdm(persona_types):
        if not os.path.isfile(
            f"results/{args.model.split('/')[-1]}_{persona_type}_{'long_' if args.long else ''}results.pkl"
        ):
            continue
        if not args.long:
            # with open(
            #     f"/scratch/vneplen/user-identity/{args.model.split('/')[-1]}_{persona_type}_results.pkl",
            #     "rb",
            # ) as infile:
            #     df_pt = CPUUnpickler(infile).load()
            df_pt = pd.read_pickle(
                f"results/{args.model.split('/')[-1]}_{persona_type}_results.pkl"
            )
            print(df_pt)

            # if "model_answer_constrained_yes" in df_pt:
            #     df_pt["model_answer_constrained_yes"] = df_pt[
            #         "model_answer_constrained_yes"
            #     ].map(lambda x: x.cpu().numpy())
            #     df_pt["model_answer_constrained_no"] = df_pt[
            #         "model_answer_constrained_no"
            #     ].map(lambda x: x.cpu().numpy())
            #     df_pt.loc[
            #         df_pt["model_answer_constrained_no"]
            #         > df_pt["model_answer_constrained_yes"],
            #         "model_answer_constrained",
            #     ] = "No"
            #     df_pt.loc[
            #         df_pt["model_answer_constrained_yes"]
            #         > df_pt["model_answer_constrained_no"],
            #         "model_answer_constrained",
            #     ] = "Yes"

            #     df_pt = df_pt.drop(
            #         columns=[
            #             "model_answer_constrained_no",
            #             "model_answer_constrained_yes",
            #         ]
            #     )
            #     df_pt = df_pt.astype(object)
            #     df_pt.to_pickle(
            #         f"/scratch/vneplen/user-identity/{args.model.split('/')[-1]}_{persona_type}_results.pkl"
            #     )
            # df_pt = df_pt.drop(columns=["prompt_text", "persona_text"])

        else:
            df_pt = pd.read_pickle(
                f"results/{args.model.split('/')[-1]}_{persona_type}_long_results.pkl"
            )
        df = pd.concat([df, df_pt], ignore_index=True)
    df.to_pickle(
        f"results/{args.model.split('/')[-1]}_{'long_' if args.long else ''}results.pkl"
    )
