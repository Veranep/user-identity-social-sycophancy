import argparse
import pandas as pd
from scipy.stats import f_oneway, tukey_hsd
from transformers import pipeline

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
    if os.path.isfile(
        f"results/{args.model.split('/')[1]}_results_emotions.pkl"
    ):
        df = pd.read_pickle(
            f"results/{args.model.split('/')[1]}_results_emotions.pkl"
        )
    else:
        df = pd.read_pickle(f"results/{args.model.split('/')[1]}_results.pkl")
        emotion_classifier = pipeline(
            task="text-classification",
            model="AnasAlokla/multilingual_go_emotions_V1.2",
            top_k=None,
            max_length=512,
            truncation=True,
        )
        results = emotion_classifier(df["model_answer"].tolist(), batch_size=8)
        for result in results:
            for e in result:
                emotions[e["label"]].append(e["score"])
        df_emotions = pd.DataFrame(emotions)
        df = pd.concat([df, df_emotions], axis=1)
        df.to_pickle(
            f"results/{args.model.split('/')[1]}_results_emotions.pkl"
        )

    for emotion in list(emotions.keys()):
        for var in ["persona_type", "persona_id", "persona_text"]:
            values = []
            for v in df[var].unique():
                if pd.isna(v):
                    vals = 1 * (df.loc[df[var].isna(), emotion])
                else:
                    vals = 1 * (df.loc[df[var] == v, emotion])
                values.append(vals.tolist())
            pvalue = f_oneway(*values).pvalue
            if pvalue < 0.05:
                print(emotion, var, tukey_hsd(*values))
                print(df.groupby(var, dropna=False)[emotion].mean())
