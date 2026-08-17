import argparse
from collections import Counter
from marked_words import (
    get_marked_words_short,
    get_marked_words_long_advice,
    get_marked_words_long_history,
)
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
from tqdm import tqdm
from wordcloud import WordCloud

models = [
    "Qwen3.6-35B-A3B",
    "gemma-3-12b-it",
    "gemma-3-27b-it",
    "Llama-3.1-8B-Instruct",
    "Qwen3.6-27B",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--long", action="store_true", help="Whether to have long replies"
    )
    args = parser.parse_args()
    if not args.long and os.path.isfile(f"data/marked_words_least.pkl"):
        with open(
            f"data/marked_words_least{'_long' if args.long else ''}.pkl", "rb"
        ) as infile:
            least_sycophantic = pickle.load(infile)
        with open(
            f"data/marked_words_most{'_long' if args.long else ''}.pkl", "rb"
        ) as infile:
            most_sycophantic = pickle.load(infile)

        print("Least sycophantic")
        print(least_sycophantic.most_common(20))
        print("Most sycophantic")
        print(most_sycophantic.most_common(20))
        # cloud_least = WordCloud(
        #     background_color="white"
        # ).generate_from_frequencies(least_sycophantic)
        # plt.imshow(cloud_least, interpolation="bilinear")
        # plt.axis("off")
        # plt.savefig("figures/least_sycophantic.png", dpi=600)
        # cloud_most = WordCloud(
        #     background_color="white"
        # ).generate_from_frequencies(most_sycophantic)
        # plt.imshow(cloud_most, interpolation="bilinear")
        # plt.axis("off")
        # plt.savefig("figures/most_sycophantic.png", dpi=600)
    elif args.long and os.path.isfile(
        "data/marked_words_long_history_least.pkl"
    ):
        with open("data/marked_words_long_advice_least.pkl", "rb") as infile:
            least_sycophantic = pickle.load(infile)
        with open("data/marked_words_long_advice_most.pkl", "rb") as infile:
            most_sycophantic = pickle.load(infile)

        print("Least sycophantic")
        print(least_sycophantic.most_common(20))
        print("Most sycophantic")
        print(most_sycophantic.most_common(20))

        with open("data/marked_words_long_history_least.pkl", "rb") as infile:
            least_negative = pickle.load(infile)
        with open("data/marked_words_long_history_most.pkl", "rb") as infile:
            most_negative = pickle.load(infile)

        print("Least negative")
        print(least_negative.most_common(20))
        print("Most negative")
        print(most_negative.most_common(20))
    else:
        if args.long:
            least_sycophantic_advice = Counter()
            most_sycophantic_advice = Counter()
            for model in tqdm(models):
                if not os.path.isfile(f"results/{model}_long_results.pkl"):
                    continue
                df_long = pd.read_pickle(f"results/{model}_long_results.pkl")
                df_short = pd.read_pickle(f"results/{model}_results.pkl")
                least_s, most_s = get_marked_words_long_advice(
                    df_short, df_long, 200
                )
                del df_long, df_short
                least_s = [t[0] for t in least_s]
                most_s = [t[0] for t in most_s]
                least_sycophantic_advice.update(least_s)
                most_sycophantic_advice.update(most_s)
            with open(
                "data/marked_words_long_advice_least.pkl", "wb"
            ) as outfile:
                pickle.dump(least_sycophantic_advice, outfile)
            with open(
                "data/marked_words_long_advice_most.pkl", "wb"
            ) as outfile:
                pickle.dump(most_sycophantic_advice, outfile)

            least_negative_advice = Counter()
            most_negative_advice = Counter()
            df_personas = pd.read_pickle("data/base_data_persona.gz")
            for model in tqdm(models):
                if not os.path.isfile(
                    f"results/{model}_long_results_sentiment.pkl"
                ):
                    continue
                df_long_sent = pd.read_pickle(
                    f"results/{model}_long_results_sentiment.pkl"
                )
                least_n, most_n = get_marked_words_long_history(
                    df_long_sent, df_personas, 200
                )
                del df_long_sent
                least_n = [t[0] for t in least_n]
                most_n = [t[0] for t in most_n]
                least_negative_advice.update(least_n)
                most_negative_advice.update(most_n)

            del df_personas
            with open(
                "data/marked_words_long_history_least.pkl", "wb"
            ) as outfile:
                pickle.dump(least_negative_advice, outfile)
            with open(
                "data/marked_words_long_history_most.pkl", "wb"
            ) as outfile:
                pickle.dump(most_negative_advice, outfile)

        else:
            least_sycophantic = Counter()
            most_sycophantic = Counter()
            df_personas = pd.read_pickle("data/base_data_persona.gz")
            for model in tqdm(models):
                df = pd.read_pickle(f"results/{model}_results.pkl")
                least_s, most_s = get_marked_words_short(df, df_personas, 200)
                least_s = [t[0] for t in least_s]
                most_s = [t[0] for t in most_s]
                least_sycophantic.update(least_s)
                most_sycophantic.update(most_s)
                del df
            del df_personas
            with open("data/marked_words_least.pkl", "wb") as outfile:
                pickle.dump(least_sycophantic, outfile)
            with open("data/marked_words_most.pkl", "wb") as outfile:
                pickle.dump(most_sycophantic, outfile)
