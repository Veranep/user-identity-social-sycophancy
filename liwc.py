import argparse
import os
from pyliwc import Liwc
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
        f"results/{args.model.split('/')[-1]}_long_results_liwc.pkl"
    ):
        df = pd.read_pickle(
            f"results/{args.model.split('/')[-1]}_long_results_liwc.pkl"
        )
    else:
        df = pd.read_pickle(
            f"results/{args.model.split('/')[-1]}_long_results.pkl"
        )
        liwc = Liwc("/opt/liwc-22/bin/LIWC-22-cli")
        liwc_dict = "LIWC22"
        df = pd.concat(
            [
                df,
                liwc.analyze_df(
                    df["model_answer"], liwc_dict=liwc_dict
                ).add_prefix("liwc_"),
            ],
            axis=1,
        )
        df.to_pickle(
            f"results/{args.model.split('/')[-1]}_long_results_liwc.pkl"
        )
    for liwc_feature in [c for c in df.columns if c.startswith("liwc_")]:
        values = []
        for v in df["persona_type"].unique():
            if pd.isna(v):
                vals = 1 * (df.loc[df["persona_type"].isna(), liwc_feature])
            else:
                vals = 1 * (df.loc[df["persona_type"] == v, liwc_feature])
            values.append(vals.tolist())
        pvalue = f_oneway(*values).pvalue
        if pvalue < 0.05:
            print(liwc_feature, "persona_type", tukey_hsd(*values))
            print(
                df.groupby("persona_type", dropna=False)[liwc_feature].mean()
            )
            print(
                df.groupby("persona_text", dropna=False)[liwc_feature].mean()
            )
            p_ids = (
                df.groupby("persona_id", dropna=False)[liwc_feature]
                .mean()
                .sort_values()
                .reset_index()
            )
            largest = p_ids.nlargest(5, liwc_feature)["persona_id"].values
            smallest = p_ids.nsmallest(5, liwc_feature)["persona_id"].values
            print("Largest:")
            for p_id in largest:
                print(df[df["persona_id"] == p_id]["persona_text"].unique())
            print("Smallest:")
            for p_id in smallest:
                print(df[df["persona_id"] == p_id]["persona_text"].unique())
