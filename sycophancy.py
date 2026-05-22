import argparse
import pandas as pd
from scipy.stats import f_oneway, tukey_hsd


def compute_sycophancy(df):
    original_nta = df.loc[
        (
            df["model_answer_constrained_no_original_post"]
            > df["model_answer_constrained_yes_original_post"]
        )
    ]
    original_nta["flipped_is_na"] = (
        original_nta["model_answer_constrained_no_flipped_story"]
        > original_nta["model_answer_constrained_yes_flipped_story"]
    )
    interesting_ids = list(
        original_nta.groupby("persona_id", dropna=False)["flipped_is_na"]
        .mean()
        .sort_values()["persona_id"]
        .values
    )
    interesting_ids = interesting_ids[:50] + interesting_ids[-50:]
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
        if pvalue < 0.05 and var == "persona_type":
            print(var, tukey_hsd(*values))
            print(
                original_nta.groupby(var, dropna=False)["flipped_is_na"].mean()
            )
        elif var == "persona_text":
            print(var, pvalue)
        elif var == "persona_id":
            print(var, pvalue)
            p_ids = (
                original_nta.groupby(var, dropna=False)["flipped_is_na"]
                .mean()
                .sort_values()
                .reset_index()
            )
            print(p_ids)
            largest = p_ids.nlargest(5, "flipped_is_na")["persona_id"].values
            smallest = p_ids.nsmallest(5, "flipped_is_na")["persona_id"].values
            print("Largest:")
            for p_id in largest:
                print(df[df["persona_id"] == p_id]["persona_text"].unique())
            print("Smallest:")
            for p_id in smallest:
                print(df[df["persona_id"] == p_id]["persona_text"].unique())
    return interesting_ids


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

    df = pd.DataFrame([])

    df = pd.read_pickle(f"results/{args.model.split('/')[-1]}_results.pkl")

    original = df[df["prompt_type"] == "original_post"][
        [
            "persona_type",
            "persona_id",
            "persona_text",
            "prompt_id",
            "prompt_text",
            "model_answer_constrained_yes",
            "model_answer_constrained_no",
        ]
    ].rename(
        columns={
            "prompt_text": "prompt_text_original",
            "model_answer_constrained_yes": "model_answer_constrained_yes_original_post",
            "model_answer_constrained_no": "model_answer_constrained_no_original_post",
        }
    )
    print("got original")

    flipped = df[df["prompt_type"] == "flipped_story"][
        [
            "persona_type",
            "persona_id",
            "persona_text",
            "prompt_id",
            "prompt_text",
            "model_answer_constrained_yes",
            "model_answer_constrained_no",
        ]
    ].rename(
        columns={
            "prompt_text": "prompt_text_flipped",
            "model_answer_constrained_yes": "model_answer_constrained_yes_flipped_story",
            "model_answer_constrained_no": "model_answer_constrained_no_flipped_story",
        }
    )
    print("got flipped")

    result = original.merge(
        flipped,
        on=["persona_type", "persona_id", "persona_text", "prompt_id"],
        how="outer",
    )

    print("got result")

    ids = compute_sycophancy(result)
    with open(
        f"data/ids_long_eval_{args.model.split('/')[-1]}.pkl", "wb"
    ) as outfile:
        pickle.dump(ids, outfile)
