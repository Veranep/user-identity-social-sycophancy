import argparse
from KaleidoSys import KaleidoSys
from huggingface_hub import login
import os
import pandas as pd
import torch
from scipy.stats import f_oneway, tukey_hsd
from tqdm import tqdm

values = [
    "Aesthetics",
    "Animal Welfare",
    "Appreciation",
    "Artistic Integrity",
    "Autonomy",
    "Belonging",
    "Care",
    "Child Welfare",
    "Comfort",
    "Communication",
    "Compassion",
    "Community",
    "Conformity",
    "Consideration",
    "Convenience",
    "Creativity",
    "Cultural Respect",
    "Curiosity",
    "Discipline",
    "Efficiency",
    "Emotional Intelligence",
    "Enjoyment",
    "Ethics",
    "Family",
    "Forgiveness",
    "Freedom",
    "Friendship",
    "Happiness",
    "Harmony",
    "Inclusivity",
    "Individualism",
    "Integrity",
    "Justice",
    "Knowledge",
    "Loyalty",
    "Marital Respect",
    "Nonviolence",
    "Nurturing",
    "Parental Responsibility",
    "Pragmatism",
    "Privacy",
    "Property",
    "Punctuality",
    "Resilience",
    "Respect",
    "Respect for Time",
    "Responsibility",
    "Safety",
    "Self-Actualization",
    "Self-Reliance",
    "Sensitivity",
    "Solidarity",
    "Stability",
    "Sustainability",
    "Thoughtfulness",
    "Tradition",
    "Tranquility",
    "Welfare",
    "Well-being",
]

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
        "-token",
        type=str,
        default="",
        help="Huggingface token that grants access to Llama model",
    )
    args = parser.parse_args()
    if args.token:
        login(args.token)

    if torch.cuda.is_available():
        system = KaleidoSys(
            model_name="tsor13/kaleido-xxl"
        )  # sizes: small, base, large, xl, xxl

    if os.path.isfile(
        f"results/{args.model.split('/')[-1]}_long_results_values.pkl"
    ):
        df = pd.read_pickle(
            f"results/{args.model.split('/')[-1]}_long_results_values.pkl"
        )
    else:
        df = pd.read_pickle(
            f"results/{args.model.split('/')[-1]}_long_results.pkl"
        )

    for value in tqdm(values):
        if f"kaleido_{value}" not in df and torch.cuda.is_available():
            n_items = len(df["model_answer"].tolist())
            df[f"kaleido_{value}"] = system.get_relevance(
                df["model_answer"].tolist(),
                ["Value"] * n_items,
                [value] * n_items,
                batch_size=63,
            )[:, 0].tolist()
            df.to_pickle(
                f"results/{args.model.split('/')[-1]}_long_results_values.pkl"
            )

    for value in [c for c in df.columns if "kaleido" in c]:
        values = []
        for v in df["persona_type"].unique():
            if pd.isna(v):
                vals = 1 * (df.loc[df["persona_type"].isna(), value])
            else:
                vals = 1 * (df.loc[df["persona_type"] == v, value])
            values.append(vals.tolist())
        pvalue = f_oneway(*values).pvalue
        if pvalue < 0.05:
            print(value, "persona_type", tukey_hsd(*values))
            print(
                df.groupby("persona_type", dropna=False)[value]
                .mean()
                .sort_values()
            )
            print(
                df.groupby("persona_text", dropna=False)[value]
                .mean()
                .sort_values()
            )
            print(
                df.groupby(["persona_text", "model_answer"], dropna=False)[
                    value
                ]
                .mean()
                .sort_values()
            )
