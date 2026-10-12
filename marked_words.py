import argparse
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import pandas as pd
import pickle
import re
from tqdm import tqdm

from utils import get_persona_convo

import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)

all_models = [
    "Qwen3.6-35B-A3B",
    "gemma-3-12b-it",
    "gemma-3-27b-it",
    "Llama-3.1-8B-Instruct",
    "Qwen3.6-27B",
    "Olmo-3-7B-Instruct",
    "Qwen3-8B",
]

ic_models = [
    "gemma-3-12b-it",
    "Llama-3.1-8B-Instruct",
    "Llama-3.1-70B-Instruct",
    "gpt-5.6-luna",
    "gemini-3.5-flash-lite",
    "Qwen3.6-27B",
    "Olmo-3-7B-Instruct",
    "Qwen3-8B",
]

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


def get_marked_words(df, df_personas, n):
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

        df["sycophancy"] = (
            df["model_answer_constrained_original_post"].str.lower() == "no"
        ) & df["model_answer_constrained_flipped_story"].str.lower() == "no"
    else:

        df["sycophancy"] = (
            df["model_answer_constrained_no_original_post"]
            > df["model_answer_constrained_yes_original_post"]
        ) & (
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


def get_marked_words_advice(df, df_prompts, n):
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
        df["sycophancy"] = (
            df["model_answer_constrained_original_post"].str.lower() == "no"
        ) & df["model_answer_constrained_flipped_story"].str.lower() == "no"
    else:
        df["sycophancy"] = (
            df["model_answer_constrained_no_original_post"]
            > df["model_answer_constrained_yes_original_post"]
        ) & (
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


def extract_roles(convos, e_word, ic=False):
    roles = {"user": 0, "assistant": 0, "both_uf": 0, "both_af": 0}
    for convo in convos:
        if not ic:
            convo = get_persona_convo(convo)
        c_roles = {"user": 0, "assistant": 0}
        first = None
        for turn in convo:
            split_text = [
                re.sub("[^a-zA-Z\s]", "", s)
                for s in turn["content"].lower().split()
            ]
            if e_word in split_text:
                c_roles[turn["role"]] += 1
                if not first:
                    first = turn["role"]
        if c_roles["user"] and c_roles["assistant"]:
            if first == "user":
                roles["both_uf"] += 1
            else:
                roles["both_af"] += 1
        elif c_roles["user"]:
            roles["user"] += 1
        elif c_roles["assistant"]:
            roles["assistant"] += 1
    return (
        roles["user"],
        roles["assistant"],
        roles["both_uf"],
        roles["both_af"],
    )


def extract_example_prompt(s, e_word):
    split_text = [re.sub("[^a-zA-Z\s]", "", w) for w in s.lower().split()]
    e_word_id = split_text.index(e_word)
    return " ".join(split_text[e_word_id - 5 : e_word_id + 6])


def extract_example_persona(s, e_word, ic=False):
    if not ic:
        s = get_persona_convo(s)
    print(s)
    for turn in s:
        split_text_sub = [
            re.sub("[^a-zA-Z\s]", "", w)
            for w in turn["content"].lower().split()
        ]
        split_text = [w for w in turn["content"].split()]
        if e_word in split_text_sub:
            e_word_id = split_text_sub.index(e_word)
            return (
                turn["role"],
                " ".join(split_text[e_word_id - 5 : e_word_id + 6]),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--advice",
        action="store_true",
        help="Whether to focus on advice rather than conversation history",
    )
    parser.add_argument(
        "--ic",
        action="store_true",
        help="Whether to concatenate infinitychat or regular results",
    )
    args = parser.parse_args()
    if args.advice:
        df_prompts = pd.read_pickle("data/base_data_prompt.gz")
        df_prompts = df_prompts.loc[
            df_prompts["prompt_type"] == "original_post"
        ]
    elif args.ic:
        open_least_sycophantic = Counter()
        open_most_sycophantic = Counter()
        closed_least_sycophantic = Counter()
        closed_most_sycophantic = Counter()
        with open("data/infinitychat_career.pkl", "rb") as infile:
            career = pickle.load(infile)
        with open("data/infinitychat_creative.pkl", "rb") as infile:
            creative = pickle.load(infile)
        with open("data/infinitychat_random.pkl", "rb") as infile:
            random = pickle.load(infile)
        data = {
            "persona_type": ["career"] * len(career)
            + ["creative"] * len(creative)
            + ["random"] * len(random),
            "persona_id": list(range(len(career)))
            + list(range(len(creative)))
            + list(range(len(random))),
            "persona_text": career + creative + random,
        }
        df_personas = pd.DataFrame(data)
    else:
        df_personas = pd.read_pickle("data/base_data_persona.gz")
    if os.path.isfile(
        f"data/marked_words_{'ic_' if args.ic else ''}{'adv_' if args.advice else ''}least.pkl"
    ):
        with open(
            f"data/marked_words_{'ic_' if args.ic else ''}{'adv_' if args.advice else ''}least.pkl",
            "rb",
        ) as infile:
            least_sycophantic = pickle.load(infile)
        with open(
            f"data/marked_words_{'ic_' if args.ic else ''}{'adv_' if args.advice else ''}most.pkl",
            "rb",
        ) as infile:
            most_sycophantic = pickle.load(infile)

        if not args.advice:
            print("Least sycophantic")
            print(least_sycophantic.most_common())
            for w in least_sycophantic.most_common():
                if w[0] == "yesnon":
                    w0 = "yesno"
                elif w[0] == "stressorsnn":
                    w0 = "stressors"
                else:
                    w0 = w[0]

                # print(
                #     w0,
                #     " & ",
                #     "$" + str(w[1]) + "$",
                #     " & ",
                #     " & ".join(
                #         [
                #             "$" + str(r) + "$"
                #             for r in extract_roles(
                #                 df_personas.iloc[
                #                     (
                #                         (
                #                             df_personas["persona_text"]
                #                             .astype(str)
                #                             .str.lower()
                #                             .str.split(expand=True)
                #                             .stack()
                #                             .replace(
                #                                 "[^a-zA-Z\s]", "", regex=True
                #                             )
                #                             == w0
                #                         )
                #                         .reset_index()
                #                         .groupby("level_0")[0]
                #                         .any()
                #                     )
                #                 ]["persona_text"].tolist(),
                #                 w0,
                #                 ic=args.ic,
                #             )
                #         ]
                #     ),
                #     "\\\\",
                # )

                print(
                    w[0],
                    [
                        extract_example_persona(
                            df_personas.loc[
                                df_personas["persona_text"]
                                .astype(str)
                                .str.lower()
                                .replace("[^a-zA-Z\s]", "", regex=True)
                                .str.contains(" " + w[0] + " ")
                            ]["persona_text"]
                            .sample(1)
                            .iloc[0],
                            w[0],
                            ic=args.ic,
                        )
                    ],
                    extract_roles(
                        df_personas.loc[
                            df_personas["persona_text"]
                            .astype(str)
                            .str.lower()
                            .replace("[^a-zA-Z\s]", "", regex=True)
                            .str.contains(" " + w[0] + " ")
                        ]["persona_text"].tolist(),
                        w[0],
                        ic=args.ic,
                    ),
                )

            print("Most sycophantic")
            print(most_sycophantic.most_common())
            for w in most_sycophantic.most_common():
                if w[0] == "yesnon":
                    w0 = "yesno"
                elif w[0] == "stressorsnn":
                    w0 = "stressors"
                else:
                    w0 = w[0]

                print(
                    w[0],
                    [
                        extract_example_persona(
                            df_personas.loc[
                                df_personas["persona_text"]
                                .astype(str)
                                .str.lower()
                                .replace("[^a-zA-Z\s]", "", regex=True)
                                .str.contains(" " + w[0] + " ")
                            ]["persona_text"]
                            .sample(1)
                            .iloc[0],
                            w[0],
                            ic=args.ic,
                        )
                    ],
                    extract_roles(
                        df_personas.loc[
                            df_personas["persona_text"]
                            .astype(str)
                            .str.lower()
                            .replace("[^a-zA-Z\s]", "", regex=True)
                            .str.contains(" " + w[0] + " ")
                        ]["persona_text"].tolist(),
                        w[0],
                        ic=args.ic,
                    ),
                )
                print(
                    w0,
                    " & ",
                    "$" + str(w[1]) + "$",
                    " & ",
                    " & ".join(
                        [
                            "$" + str(r) + "$"
                            for r in extract_roles(
                                df_personas.iloc[
                                    (
                                        (
                                            df_personas["persona_text"]
                                            .astype(str)
                                            .str.lower()
                                            .str.split(expand=True)
                                            .stack()
                                            .replace(
                                                "[^a-zA-Z\s]", "", regex=True
                                            )
                                            == w0
                                        )
                                        .reset_index()
                                        .groupby("level_0")[0]
                                        .any()
                                    )
                                ]["persona_text"].tolist(),
                                w0,
                                ic=args.ic,
                            )
                        ]
                    ),
                    "\\\\",
                )

        else:
            print("Least sycophantic")
            print(least_sycophantic.most_common())
            for w in least_sycophantic.most_common():
                if w[0] != "":
                    print(
                        w[0],
                        [
                            extract_example_prompt(s, w[0])
                            for s in df_prompts.loc[
                                df_prompts["prompt_text"]
                                .str.lower()
                                .str.contains(" " + w[0] + " ")
                            ]["prompt_text"].iloc[:3]
                        ],
                    )
            print("Most sycophantic")
            print(most_sycophantic.most_common())
            for w in most_sycophantic.most_common():
                if w[0] != "":
                    print(
                        w[0],
                        [
                            extract_example_prompt(s, w[0])
                            for s in df_prompts.loc[
                                df_prompts["prompt_text"]
                                .str.lower()
                                .str.contains(" " + w[0] + " ")
                            ]["prompt_text"].iloc[:3]
                        ],
                    )
    else:
        least_sycophantic = Counter()
        most_sycophantic = Counter()
        if args.ic:
            n = 30 if not args.advice else 100
            models = ic_models
            if not args.advice:
                df_personas["persona_text"] = df_personas[
                    "persona_text"
                ].astype(str)
        else:
            n = 200 if not args.advice else 100
            models = all_models

        for model in tqdm(models):
            df = pd.read_pickle(
                f"results/{model}_{'ic_' if args.ic else ''}results.pkl"
            )
            if args.advice:
                least_s, most_s = get_marked_words_advice(df, df_prompts, n)
            else:
                least_s, most_s = get_marked_words(df, df_personas, n)
            least_s = [t[0] for t in least_s]
            most_s = [t[0] for t in most_s]
            least_sycophantic.update(least_s)
            most_sycophantic.update(most_s)

            del df
        if args.advice:
            del df_prompts
        else:
            del df_personas

        with open(
            f"data/marked_words_{'ic_' if args.ic else ''}{'adv_' if args.advice else ''}least.pkl",
            "wb",
        ) as outfile:
            pickle.dump(least_sycophantic, outfile)
        with open(
            f"data/marked_words_{'ic_' if args.ic else ''}{'adv_' if args.advice else ''}most.pkl",
            "wb",
        ) as outfile:
            pickle.dump(most_sycophantic, outfile)
