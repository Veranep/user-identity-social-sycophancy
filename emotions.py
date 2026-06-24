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
        help="Whether to evaluate emotions in LLM-generated advice or in the identity-conveying conversation history",
    )
    args = parser.parse_args()
    emotions = {
        "admiration": [],
        "amusement": [],
        "anger": [],
        "annoyance": [],
        "approval": [],
        "caring": [],
        "confusion": [],
        "curiosity": [],
        "desire": [],
        "disappointment": [],
        "disapproval": [],
        "disgust": [],
        "embarrassment": [],
        "excitement": [],
        "fear": [],
        "gratitude": [],
        "grief": [],
        "joy": [],
        "love": [],
        "nervousness": [],
        "optimism": [],
        "pride": [],
        "realization": [],
        "relief": [],
        "remorse": [],
        "sadness": [],
        "surprise": [],
        "neutral": [],
    }
    if args.type == "advice":
        col = "model_answer"
        save_file = (
            f"results/{args.model.split('/')[-1]}_long_results_emotions.pkl"
        )
        if os.path.isfile(save_file):
            df = pd.read_pickle(save_file)
            results = True
            df_personas = pd.read_pickle("data/base_data_persona.gz")
            df = df.merge(df_personas, on=["persona_id", "persona_type"])
            del df_personas
        else:
            df = pd.read_pickle(
                f"results/{args.model.split('/')[-1]}_long_results.pkl"
            )
            results = False

    else:
        col = "content"
        save_file = "data/base_data_persona_turns_emotions.gz"
        if os.path.isfile(save_file):
            df = pd.read_pickle(save_file)
            results = True
        else:
            df = pd.read_pickle("data/base_data_persona_turns.gz")
            results = False

    if not results:
        emotion_classifier = pipeline(
            task="text-classification",
            model="AnasAlokla/multilingual_go_emotions_V1.2",
            top_k=None,
            max_length=512,
            truncation=True,
        )
        results = [
            result
            for result in tqdm(
                emotion_classifier(
                    ListDataset(df[col].tolist()), batch_size=32
                )
            )
        ]
        for result in results:
            for e in result:
                emotions[e["label"]].append(e["score"])
        df_emotions = pd.DataFrame(emotions)
        df = pd.concat([df, df_emotions], axis=1)
        df.to_pickle(save_file)

    else:
        for emotion in list(emotions.keys()):
            values = []
            for v in df["persona_type"].unique():
                if pd.isna(v):
                    vals = 1 * (df.loc[df["persona_type"].isna(), emotion])
                else:
                    vals = 1 * (df.loc[df["persona_type"] == v, emotion])
                values.append(vals.tolist())
            pvalue = f_oneway(*values).pvalue
            if pvalue < 0.05:
                print(emotion, "persona_type", tukey_hsd(*values))
                print(
                    df.groupby("persona_type", dropna=False)[emotion]
                    .mean()
                    .sort_values()
                )
                # p_ids = (
                #     df.groupby("persona_id", dropna=False)[emotion]
                #     .mean()
                #     .sort_values()
                #     .reset_index()
                # )
                # largest = p_ids.nlargest(5, emotion)["persona_id"].values
                # smallest = p_ids.nsmallest(5, emotion)["persona_id"].values
                # print("Largest:")
                # for p_id in largest:
                #     print(
                #         df[df["persona_id"] == p_id]["persona_text"].unique()
                #     )
                # print("Smallest:")
                # for p_id in smallest:
                #     print(
                #         df[df["persona_id"] == p_id]["persona_text"].unique()
                #     )
                if args.type == "advice":
                    print(
                        df.groupby("persona_text", dropna=False)[emotion]
                        .mean()
                        .sort_values()
                    )
