import argparse
import os
import pandas as pd
from scipy.stats import f_oneway, tukey_hsd
from tqdm import tqdm
from transformers import pipeline
from torch.utils.data import Dataset


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
        "-t",
        "--type",
        type=str,
        choices=["advice", "conv_history"],
        help="Whether to evaluate social_dimensions in LLM-generated advice or in the identity-conveying conversation history",
    )
    args = parser.parse_args()
    social_dimensions = {
        "social_support": [],
        "similarity": [],
        "romance": [],
        "respect": [],
        "power": [],
        "knowledge": [],
        "identity": [],
        "fun": [],
        "conflict": [],
        "trust": [],
    }
    if args.type == "advice":
        col = "model_answer"
        save_file = f"results/{args.model.split('/')[-1]}_long_results_social_dimensions.pkl"
        if os.path.isfile(save_file):
            df = pd.read_pickle(save_file)
            results = True
            df_personas = pd.read_pickle("data/base_data_persona.gz")
            df = df.merge(df_personas, on=["persona_id", "persona_type"])
            del df_personas
        else:
            df = pd.read_pickle(
                f"results/{args.model.split('/')[-1]}_long_results.pkl"
            )
            results = False

    else:
        col = "content"
        save_file = "data/base_data_persona_turns_social_dimensions.gz"
        if os.path.isfile(save_file):
            df = pd.read_pickle(save_file)
            results = True
        else:
            df = pd.read_pickle("data/base_data_persona_turns.gz")
            results = False

    if not results:
        for d in social_dimensions:
            classifier = pipeline(
                task="text-classification",
                model=f"Blablablab/10dimensions-{d}",
                max_length=512,
                truncation=True,
                top_k=None,
            )
            results = [
                r["score"]
                for result in tqdm(
                    classifier(ListDataset(df[col].tolist()), batch_size=32)
                )
                for r in result
                if r["label"] == "positive"
            ]
            social_dimensions[d] = results
        df_social_dimensions = pd.DataFrame(social_dimensions)
        df = pd.concat([df, df_social_dimensions], axis=1)
        df.to_pickle(save_file)

    else:
        for d in list(social_dimensions.keys()):
            values = []
            for v in df["persona_type"].unique():
                if pd.isna(v):
                    vals = 1 * (df.loc[df["persona_type"].isna(), d])
                else:
                    vals = 1 * (df.loc[df["persona_type"] == v, d])
                values.append(vals.tolist())
            pvalue = f_oneway(*values).pvalue
            if pvalue < 0.05:
                print(d, "persona_type", tukey_hsd(*values))
                print(
                    df.groupby("persona_type", dropna=False)[d]
                    .mean()
                    .sort_values()
                )
                # p_ids = (
                #     df.groupby("persona_id", dropna=False)[d]
                #     .mean()
                #     .sort_values()
                #     .reset_index()
                # )
                # largest = p_ids.nlargest(5, d)["persona_id"].values
                # smallest = p_ids.nsmallest(5, d)["persona_id"].values
                # print("Largest:")
                # for p_id in largest:
                #     print(
                #         df[df["persona_id"] == p_id]["persona_text"].unique()
                #     )
                # print("Smallest:")
                # for p_id in smallest:
                #     print(
                #         df[df["persona_id"] == p_id]["persona_text"].unique()
                #     )
                if args.type == "advice":
                    print(
                        df.groupby("persona_text", dropna=False)[d]
                        .mean()
                        .sort_values()
                    )
