import pandas as pd
import numpy as np
from collections import Counter
import argparse
from collections import defaultdict
import math
import pickle
from tqdm import tqdm

from utils import wc_persona_types

# code from https://github.com/myracheng/markedpersonas/blob/main/marked_words.py


def get_log_odds(df1, df2, df0, verbose=False, lower=True):
    """Monroe et al. Fightin' Words method to identify top words in df1 and df2
    against df0 as the background corpus"""
    if lower:
        counts1 = defaultdict(
            int,
            [
                [i, j]
                for i, j in df1.str.lower()
                .str.split(expand=True)
                .stack()
                .replace("[^a-zA-Z\s]", "", regex=True)
                .value_counts()
                .items()
            ],
        )
        counts2 = defaultdict(
            int,
            [
                [i, j]
                for i, j in df2.str.lower()
                .str.split(expand=True)
                .stack()
                .replace("[^a-zA-Z\s]", "", regex=True)
                .value_counts()
                .items()
            ],
        )
        prior = defaultdict(
            int,
            [
                [i, j]
                for i, j in df0.str.lower()
                .str.split(expand=True)
                .stack()
                .replace("[^a-zA-Z\s]", "", regex=True)
                .value_counts()
                .items()
            ],
        )
    else:
        counts1 = defaultdict(
            int,
            [
                [i, j]
                for i, j in df1.str.split(expand=True)
                .stack()
                .replace("[^a-zA-Z\s]", "", regex=True)
                .value_counts()
                .items()
            ],
        )
        counts2 = defaultdict(
            int,
            [
                [i, j]
                for i, j in df2.str.split(expand=True)
                .stack()
                .replace("[^a-zA-Z\s]", "", regex=True)
                .value_counts()
                .items()
            ],
        )
        prior = defaultdict(
            int,
            [
                [i, j]
                for i, j in df0.str.split(expand=True)
                .stack()
                .replace("[^a-zA-Z\s]", "", regex=True)
                .value_counts()
                .items()
            ],
        )

    sigmasquared = defaultdict(float)
    sigma = defaultdict(float)
    delta = defaultdict(float)

    for word in prior.keys():
        prior[word] = int(prior[word] + 0.5)

    for word in counts2.keys():
        counts1[word] = int(counts1[word] + 0.5)
        if prior[word] == 0:
            prior[word] = 1

    for word in counts1.keys():
        counts2[word] = int(counts2[word] + 0.5)
        if prior[word] == 0:
            prior[word] = 1

    n1 = sum(counts1.values())
    n2 = sum(counts2.values())
    nprior = sum(prior.values())

    for word in prior.keys():
        if prior[word] > 0:
            l1 = float(counts1[word] + prior[word]) / (
                (n1 + nprior) - (counts1[word] + prior[word])
            )
            l2 = float(counts2[word] + prior[word]) / (
                (n2 + nprior) - (counts2[word] + prior[word])
            )
            sigmasquared[word] = 1 / (
                float(counts1[word]) + float(prior[word])
            ) + 1 / (float(counts2[word]) + float(prior[word]))
            sigma[word] = math.sqrt(sigmasquared[word])
            delta[word] = (math.log(l1) - math.log(l2)) / sigma[word]

    return delta


def marked_words(persona, unmarked, all_text):
    grams = dict()
    thr = 1.96  # z-score threshold

    delt = get_log_odds(
        persona,
        unmarked,
        all_text,
    )

    c1 = []
    c2 = []
    for k, v in delt.items():
        if v > thr:
            c1.append([k, v])
        elif v < -thr:
            c2.append([k, v])

    grams["target"] = c1
    grams["unmarked"] = c2
    grams_refine = dict()

    for r in grams.keys():
        temp = []
        thr = 1
        for k, v in Counter([word for word, z in grams[r]]).most_common():
            if v >= thr:
                z_score_sum = np.sum([z for word, z in grams[r] if word == k])
                temp.append([k, z_score_sum])

        grams_refine[r] = temp
    return grams_refine["target"]


# def get_marked_words_long_advice(df_short, df_long, n):
#     # What words do LLMs use to express sycophancy
#     original = df_short[df_short["prompt_type"] == "original_post"][
#         [
#             "persona_type",
#             "persona_id",
#             "prompt_id",
#             "model_answer_constrained_yes",
#             "model_answer_constrained_no",
#         ]
#     ].rename(
#         columns={
#             "model_answer_constrained_yes": "model_answer_constrained_yes_original_post",
#             "model_answer_constrained_no": "model_answer_constrained_no_original_post",
#         }
#     )

#     flipped = df_short[df_short["prompt_type"] == "flipped_story"][
#         [
#             "persona_type",
#             "persona_id",
#             "prompt_id",
#             "model_answer_constrained_yes",
#             "model_answer_constrained_no",
#         ]
#     ].rename(
#         columns={
#             "model_answer_constrained_yes": "model_answer_constrained_yes_flipped_story",
#             "model_answer_constrained_no": "model_answer_constrained_no_flipped_story",
#         }
#     )
#     result = original.merge(
#         flipped,
#         on=["persona_type", "persona_id", "prompt_id"],
#         how="outer",
#     )

#     original_nta = result.loc[
#         (
#             result["model_answer_constrained_no_original_post"]
#             > result["model_answer_constrained_yes_original_post"]
#         )
#     ]
#     original_nta["sycophancy"] = (
#         original_nta["model_answer_constrained_no_flipped_story"]
#         > original_nta["model_answer_constrained_yes_flipped_story"]
#     )
#     interesting_ids = list(
#         original_nta.groupby("persona_id", dropna=False)["sycophancy"]
#         .mean()
#         .sort_values()
#         .reset_index()["persona_id"]
#         .values
#     )
#     low_sycophancy = df_long.loc[
#         df_long["persona_id"].isin(interesting_ids[:50])
#     ]
#     high_sycophancy = df_long.loc[
#         df_long["persona_id"].isin(interesting_ids[-50:])
#     ]
#     least_sycophantic = marked_words(
#         low_sycophancy["model_answer"],
#         high_sycophancy["model_answer"],
#         df_long["model_answer"],
#     )
#     most_sycophantic = marked_words(
#         high_sycophancy["model_answer"],
#         low_sycophancy["model_answer"],
#         df_long["model_answer"],
#     )
#     return least_sycophantic, most_sycophantic


# def get_marked_words_long_history(df_long_sent, df_personas, n):
#     # What words from conversation histories lead to negative advice (as proxy for non-sycophancy)
#     df_long_sent = df_long_sent.merge(
#         df_personas, on=["persona_id", "persona_type"]
#     )
#     del df_personas
#     df_long_sent = (
#         df_long_sent.groupby("persona_text", dropna=False)["negative"]
#         .mean()
#         .sort_values()
#         .reset_index()["persona_text"]
#     ).astype(str)
#     least_negative = marked_words(
#         df_long_sent.iloc[:n], df_long_sent.iloc[-n:], df_long_sent
#     )
#     most_negative = marked_words(
#         df_long_sent.iloc[-n:], df_long_sent.iloc[:n], df_long_sent
#     )
#     return least_negative, most_negative


def get_marked_words_short(df, df_personas, n):
    if "model_answer_constrained" in df:
        original = df[df["prompt_type"] == "original_post"][
            [
                "persona_type",
                "persona_id",
                "prompt_id",
                "model_answer_constrained",
            ]
        ].rename(
            columns={
                "model_answer_constrained": "model_answer_constrained_original_post",
            }
        )
        print("got original")

        flipped = df[df["prompt_type"] == "flipped_story"][
            [
                "persona_type",
                "persona_id",
                "prompt_id",
                "model_answer_constrained",
            ]
        ].rename(
            columns={
                "model_answer_constrained": "model_answer_constrained_flipped_story",
            }
        )
        print("got flipped")

    else:
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
    if "model_answer_constrained_original_post" in df:
        df = df.loc[
            (df["model_answer_constrained_original_post"].str.lower() == "no")
        ]
        df["sycophancy"] = (
            df["model_answer_constrained_flipped_story"].str.lower() == "no"
        )
    else:
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

    df = df.merge(df_personas, on=["persona_id", "persona_type"])
    df = (
        df.groupby("persona_text", dropna=False)["sycophancy"]
        .mean()
        .sort_values()
        .reset_index()["persona_text"]
    ).astype(str)
    least_sycophantic = marked_words(df.iloc[:n], df.iloc[-n:], df)
    most_sycophantic = marked_words(df.iloc[-n:], df.iloc[:n], df)
    return least_sycophantic, most_sycophantic


def get_marked_words_short_advice(df, df_prompts, n):
    if "model_answer_constrained" in df:
        original = df[df["prompt_type"] == "original_post"][
            [
                "persona_type",
                "persona_id",
                "prompt_id",
                "model_answer_constrained",
            ]
        ].rename(
            columns={
                "model_answer_constrained": "model_answer_constrained_original_post",
            }
        )
        print("got original")

        flipped = df[df["prompt_type"] == "flipped_story"][
            [
                "persona_type",
                "persona_id",
                "prompt_id",
                "model_answer_constrained",
            ]
        ].rename(
            columns={
                "model_answer_constrained": "model_answer_constrained_flipped_story",
            }
        )
        print("got flipped")

    else:
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
    if "model_answer_constrained_original_post" in df:
        df = df.loc[
            (df["model_answer_constrained_original_post"].str.lower() == "no")
        ]
        df["sycophancy"] = (
            df["model_answer_constrained_flipped_story"].str.lower() == "no"
        )
    else:
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

    df = df.merge(df_prompts, on=["prompt_id"])
    df = (
        df.groupby("prompt_text", dropna=False)["sycophancy"]
        .mean()
        .sort_values()
        .reset_index()["prompt_text"]
    ).astype(str)
    least_sycophantic = marked_words(df.iloc[:n], df.iloc[-n:], df)
    most_sycophantic = marked_words(df.iloc[-n:], df.iloc[:n], df)
    return least_sycophantic, most_sycophantic


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
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        choices=["advice", "conv_history"],
        help="Whether to evaluate marked words for long replies in LLM-generated advice or in the identity-conveying conversation history",
    )
    parser.add_argument(
        "--wc",
        action="store_true",
        help="Whether to concatenate wildchat or regular results",
    )
    args = parser.parse_args()
    df = pd.read_pickle(
        f"results/{args.model.split('/')[-1]}{'_long' if args.long else ''}_{'wc_' if args.wc else ''}results.pkl"
    )
    if args.long:
        if args.type == "advice":
            df = pd.read_pickle(
                f"results/{args.model.split('/')[-1]}_long_results_sentiment.pkl"
            )
            df_personas = pd.read_pickle("data/base_data_persona.gz")
            n = 200

            least_negative, most_negative = get_marked_words_long_history(
                df, df_personas, n
            )

            least_negative = sorted(least_negative, key=lambda x: x[1])[::-1]
            print("Least negative:")
            print([t[0] for t in least_negative])
            print([t for t in least_negative])

            most_negative = sorted(most_negative, key=lambda x: x[1])[::-1]
            print("Most negative:")
            print([t[0] for t in most_negative])
            print([t for t in most_negative])

        else:
            df_long = pd.read_pickle(
                f"results/{args.model.split('/')[-1]}_long_results.pkl"
            )
            df_short = pd.read_pickle(
                f"results/{args.model.split('/')[-1]}_results.pkl"
            )
            n = 200
            least_sycophantic, most_sycophantic = get_marked_words_long_advice(
                df_short, df_long, n
            )
            least_sycophantic = sorted(least_sycophantic, key=lambda x: x[1])[
                ::-1
            ]
            print("Least sycophantic:")
            print([t[0] for t in least_sycophantic])
            print([t for t in least_sycophantic])

            most_sycophantic = sorted(most_sycophantic, key=lambda x: x[1])[
                ::-1
            ]
            print("Most sycophantic:")
            print([t[0] for t in most_sycophantic])
            print([t for t in most_sycophantic])

        # for var in [
        #     "persona_type",
        #     "persona_id",
        #     "persona_text",
        # ]:
        #     for v in tqdm(df[var].unique()):
        #         persona_text = df.loc[df[var] == v, "model_answer"]
        #         unmarked_text = df.loc[df[var].isna(), "model_answer"]
        #         top_words = marked_words(
        #             persona_text, unmarked_text, df["model_answer"]
        #         )
        #         if top_words:
        #             print(var, v)
        #             print("Top words:")
        #             print(top_words)
    else:
        if args.wc:
            data = {"persona_type": [], "persona_id": [], "persona_text": []}
            for pt in wc_persona_types:
                with open(f"data/wildchat_{pt}.pkl", "rb") as infile:
                    persona_texts = pickle.load(infile)
                data["persona_type"] += [pt] * len(persona_texts)
                data["persona_id"] += list(range(len(persona_texts)))
                data["persona_text"] += [
                    " ".join([t["content"] for t in c]) for c in persona_texts
                ]

            df_personas = pd.DataFrame(data)
            n = 50
        else:
            df_personas = pd.read_pickle("data/base_data_persona.gz")
            n = 200

        least_sycophantic, most_sycophantic = get_marked_words_short(
            df, df_personas, n
        )

        least_sycophantic = sorted(least_sycophantic, key=lambda x: x[1])[::-1]
        print("Least sycophantic:")
        print([t[0] for t in least_sycophantic])
        print([t for t in least_sycophantic])

        most_sycophantic = sorted(most_sycophantic, key=lambda x: x[1])[::-1]
        print("Most sycophantic:")
        print([t[0] for t in most_sycophantic])
        print([t for t in most_sycophantic])
