import argparse
import os
from scipy.stats import f_oneway, tukey_hsd
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to evaluate",
    )
    args = parser.parse_args()
    if os.path.isfile(
        f"results/{args.model.split('/')[-1]}_long_results_features.pkl"
    ):
        df = pd.read_pickle(
            f"results/{args.model.split('/')[-1]}_long_results_features.pkl"
        )
    else:
        df = pd.read_pickle(
            f"results/{args.model.split('/')[-1]}_long_results.pkl"
        )
        col = "model_answer"
        df[f"{col}_length"] = df[col].str.len()
        df.to_pickle(
            f"results/{args.model.split('/')[-1]}_long_results_features.pkl"
        )
    for feature in ["model_answer_length"]:
        values = []
        for v in df["persona_type"].unique():
            if pd.isna(v):
                vals = 1 * (df.loc[df["persona_type"].isna(), feature])
            else:
                vals = 1 * (df.loc[df["persona_type"] == v, feature])
            values.append(vals.tolist())
        pvalue = f_oneway(*values).pvalue
        if pvalue < 0.05:
            print(feature, "persona_type", tukey_hsd(*values))
            print(
                df.groupby("persona_type", dropna=False)[feature]
                .mean()
                .sort_values()
            )
            print(
                df.groupby("persona_text", dropna=False)[feature]
                .mean()
                .sort_values()
            )
            p_ids = (
                df.groupby("persona_id", dropna=False)[feature]
                .mean()
                .sort_values()
                .reset_index()
            )
            largest = p_ids.nlargest(5, feature)["persona_id"].values
            smallest = p_ids.nsmallest(5, feature)["persona_id"].values
            print("Largest:")
            for p_id in largest:
                print(df[df["persona_id"] == p_id]["persona_text"].unique())
            print("Smallest:")
            for p_id in smallest:
                print(df[df["persona_id"] == p_id]["persona_text"].unique())
