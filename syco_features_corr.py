import argparse
import pandas as pd
from utils import feature_cols


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
        help="Whether to evaluate correlations between sycophancy and attributes of the LLM-generated advice or the identity-conveying conversation history",
    )
    parser.add_argument(
        "-ft",
        "--feature_type",
        type=str,
        choices=["emotions", "liwc", "sentiment", "politeness"],
        help="Which attributes of the LLM-generated advice or the identity-conveying conversation history to compute correlations with",
    )
    args = parser.parse_args()
    df = pd.read_pickle(f"results/{args.model.split('/')[-1]}_results.pkl")
    if args.type == "advice":
        features = pd.read_pickle(
            f"results/{args.model.split('/')[-1]}_long_results_{args.feature_type}.pkl"
        )
    else:
        features = pd.read_pickle(
            f"data/base_data_persona_turns_{args.feature_type}.gz"
        )
    original = df[df["prompt_type"] == "original_post"][
        [
            "persona_type",
            "persona_id",
            "prompt_id",
            "model_answer_constrained_yes",
            "model_answer_constrained_no",
        ]
    ].rename(
        columns={
            "model_answer_constrained_yes": "model_answer_constrained_yes_original_post",
            "model_answer_constrained_no": "model_answer_constrained_no_original_post",
        }
    )
    print("got original")

    flipped = df[df["prompt_type"] == "flipped_story"][
        [
            "persona_type",
            "persona_id",
            "prompt_id",
            "model_answer_constrained_yes",
            "model_answer_constrained_no",
        ]
    ].rename(
        columns={
            "model_answer_constrained_yes": "model_answer_constrained_yes_flipped_story",
            "model_answer_constrained_no": "model_answer_constrained_no_flipped_story",
        }
    )
    print("got flipped")
    df = original.merge(
        flipped,
        on=["persona_type", "persona_id", "prompt_id"],
        how="outer",
    )
    df = df.loc[
        (
            df["model_answer_constrained_no_original_post"]
            > df["model_answer_constrained_yes_original_post"]
        )
    ]
    df["sycophancy"] = (
        df["model_answer_constrained_no_flipped_story"]
        > df["model_answer_constrained_yes_flipped_story"]
    )
    df = (
        df.groupby(
            [
                "persona_type",
                "persona_id",
            ]
        )["sycophancy"]
        .mean()
        .reset_index()
    )

    if args.feature_type == "politeness":
        features["politeness"] = features["politeness"].replace(
            {"polite": 2, "somewhat polite": 1, "neutral": 0, "impolite": -2}
        )
    elif args.feature_type == "emotions":
        features = features.rename(columns={"neutral": "e_neutral"})
    elif args.feature_type == "sentiment":
        features = features.rename(columns={"neutral": "s_neutral"})

    if args.type == "conv_history":
        features = (
            features.drop(columns=["content"])
            .groupby(["persona_type", "persona_id", "role"])
            .mean()
            .reset_index()
        )
        user_turns = features[features["role"] == "user"][
            ["persona_type", "persona_id"] + feature_cols[args.feature_type]
        ].rename(
            columns={
                feature: "user_" + feature
                for feature in feature_cols[args.feature_type]
            }
        )
        model_turns = features[features["role"] == "assistant"][
            [
                "persona_type",
                "persona_id",
            ]
            + feature_cols[args.feature_type]
        ].rename(
            columns={
                feature: "model_" + feature
                for feature in feature_cols[args.feature_type]
            }
        )
        features = user_turns.merge(
            model_turns, on=["persona_type", "persona_id"], how="outer"
        )

    else:
        features = (
            features.drop(
                errors="ignore",
                columns=[
                    "prompt_type",
                    "prompt_text",
                    "model_answer",
                    "prompt_id",
                    "persona_text",
                ],
            )
            .groupby(["persona_type", "persona_id"])
            .mean()
            .reset_index()
        )
    df = pd.merge(df, features, on=["persona_type", "persona_id"])

    all_features = feature_cols[args.feature_type]
    if args.type == "conv_history":
        all_features = ["user_" + feat for feat in all_features] + [
            "model_" + feat for feat in all_features
        ]
    for feature_set in [
        all_features[i : i + 10] for i in range(0, len(all_features), 10)
    ]:
        print(
            df[
                [
                    "sycophancy",
                ]
                + feature_set
            ].corr()
        )
