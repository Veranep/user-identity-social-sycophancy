import pandas as pd
import numpy as np
from collections import Counter
import argparse
from collections import defaultdict
import math

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
        f"results/{args.model.split('/')[-1]}_long_results.pkl"
    )
    for var in ["persona_type", "persona_id", "persona_text"]:
        for v in df[var].unique():
            persona_text = df.loc[df[var] == v, "model_answer"]
            unmarked_text = df.loc[df[var].isna(), "model_answer"]
            top_words = marked_words(
                persona_text, unmarked_text, df["model_answer"]
            )
            print(var, v)
            print("Top words:")
            print(top_words)
