import argparse
import os
import pandas as pd
from scipy.stats import f_oneway, tukey_hsd
from tqdm import tqdm
from transformers import pipeline
from torch.utils.data import Dataset


class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


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
        help="Whether to evaluate sentiment in LLM-generated advice or in the identity-conveying conversation history",
    )
    args = parser.parse_args()
    sentiments = {"negative": [], "neutral": [], "positive": []}
    if args.type == "advice":
        col = "model_answer"
        save_file = (
            f"results/{args.model.split('/')[-1]}_long_results_sentiment.pkl"
        )
        if os.path.isfile(save_file):
            df = pd.read_pickle(save_file)
            results = True
        else:
            df = pd.read_pickle(
                f"results/{args.model.split('/')[-1]}_long_results.pkl"
            )
            results = False

    else:
        col = "content"
        save_file = "data/base_data_persona_turns_sentiment.gz"
        if os.path.isfile(save_file):
            df = pd.read_pickle(save_file)
            results = True
        else:
            df = pd.read_pickle("data/base_data_persona_turns.gz")
            results = False

    if not results:
        sentiment_classifier = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
            top_k=None,  # To return all scores for each label
            max_length=512,
            truncation=True,
        )
        results = [
            result
            for result in tqdm(
                sentiment_classifier(
                    ListDataset(df[col].tolist()), batch_size=32
                )
            )
        ]
        for result in results:
            for s in result:
                sentiments[s["label"]].append(s["score"])
        df_sentiment = pd.DataFrame(sentiments)
        df = pd.concat([df, df_sentiment], axis=1)
        df.to_pickle(save_file)

    else:
        for sentiment in list(sentiment.keys()):
            values = []
            for v in df["persona_type"].unique():
                if pd.isna(v):
                    vals = 1 * (df.loc[df["persona_type"].isna(), sentiment])
                else:
                    vals = 1 * (df.loc[df["persona_type"] == v, sentiment])
                values.append(vals.tolist())
            pvalue = f_oneway(*values).pvalue
            if pvalue < 0.05:
                print(sentiment, "persona_type", tukey_hsd(*values))
                print(
                    df.groupby("persona_type", dropna=False)[sentiment]
                    .mean()
                    .sort_values()
                )
                p_ids = (
                    df.groupby("persona_id", dropna=False)[sentiment]
                    .mean()
                    .sort_values()
                    .reset_index()
                )
                largest = p_ids.nlargest(5, sentiment)["persona_id"].values
                smallest = p_ids.nsmallest(5, sentiment)["persona_id"].values
                print("Largest:")
                for p_id in largest:
                    print(
                        df[df["persona_id"] == p_id]["persona_text"].unique()
                    )
                print("Smallest:")
                for p_id in smallest:
                    print(
                        df[df["persona_id"] == p_id]["persona_text"].unique()
                    )
                if args.type == "advice":
                    print(
                        df.groupby("persona_text", dropna=False)[sentiment]
                        .mean()
                        .sort_values()
                    )
