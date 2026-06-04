import argparse
from KaleidoSys import KaleidoSys
from huggingface_hub import login
import os
import pandas as pd
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

    system = KaleidoSys(
        model_name="tsor13/kaleido-large"
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
        if f"kaleido_{value}" not in df:
            n_items = len(df["model_answer"].tolist())
            df[f"kaleido_{value}"] = system.get_relevance(
                df["model_answer"].tolist(),
                ["Value"] * n_items,
                [value] * n_items,
                batch_size=16,
            )[:, 0].tolist()
            df.to_pickle(
                f"results/{args.model.split('/')[-1]}_long_results_values.pkl"
            )
