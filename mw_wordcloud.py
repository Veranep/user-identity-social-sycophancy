from collections import Counter
from marked_words import get_marked_words_short
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
    if os.path.isfile("data/marked_words_least.pkl"):
        with open("data/marked_words_least.pkl", "rb") as infile:
            least_sycophantic = pickle.load(infile)
        with open("data/marked_words_most.pkl", "rb") as infile:
            most_sycophantic = pickle.load(infile)
        cloud_least = WordCloud(
            background_color="white"
        ).generate_from_frequencies(least_sycophantic)
        plt.imshow(cloud_least, interpolation="bilinear")
        plt.axis("off")
        plt.savefig("figures/least_sycophantic.png", dpi=600)
        cloud_most = WordCloud(
            background_color="white"
        ).generate_from_frequencies(most_sycophantic)
        plt.imshow(cloud_most, interpolation="bilinear")
        plt.axis("off")
        plt.savefig("figures/most_sycophantic.png", dpi=600)
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
