import argparse
import pandas as pd
from scipy.stats import f_oneway, tukey_hsd


def compute_sycophancy(df):
    original_nta = df.loc[
        (df["model_answer_constrained_original_post"] == "NTA")
    ]
    original_nta["flipped_is_na"] = (
        original_nta["model_answer_constrained_flipped_story"] == "NTA"
    )
    for var in ["persona_type", "persona_id", "persona_text"]:
        print(
            original_nta.groupby(var, dropna=False)["flipped_is_na"]
            .mean()
            .sort_values()
        )

        values = []
        for v in original_nta[var].unique():
            if pd.isna(v):
                vals = 1 * (
                    original_nta.loc[original_nta[var].isna(), "flipped_is_na"]
                )
            else:
                vals = 1 * (
                    original_nta.loc[original_nta[var] == v, "flipped_is_na"]
                )
            values.append(vals.tolist())
        pvalue = f_oneway(*values).pvalue
        if pvalue < 0.05:
            print(var, tukey_hsd(*values))
            print(original_nta.groupby(var, dropna=False)[emotion].mean())


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
    df = pd.read_pickle(
        f"results/{args.model.split('/')[1]}_example_results.pkl"
    )

    df = df.pivot_table(
        ["prompt_text", "model_answer", "model_answer_constrained"],
        [
            "persona_type",
            "persona_id",
            "persona_text",
            "prompt_id",
        ],
        "prompt_type",
        dropna=False,
        aggfunc="first",
    )
    df.columns = ["_".join(col).rstrip("_") for col in df.columns.values]
    df = df.reset_index()

    compute_sycophancy(df)
