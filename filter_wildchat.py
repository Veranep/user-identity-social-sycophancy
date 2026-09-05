import argparse
import ast
from datasets import load_dataset
from huggingface_hub import login
import os
import pandas as pd
import pickle
from tqdm import tqdm

exclude = [
    "write a",
    "create me",
    "generate",
    "story",
    "write me",
    "rewrite",
    "medical",
    "python",
    "rephrase",
    "paraphrase",
    "response must be written as",
    "add description",
    "malware",
    "create a",
    "write 3 different examples",
    "arya",
    "act as",
    "translat",
    "create caption",
    "give me a",
    "explain please",
    "alyx",
    "witchaven",
    "sfm",
    "keywords for",
    "provide me with",
    "farting",
    "vite",
    "list and enumerate",
    "hentai",
    "remedies",
    "jpop",
    "frequently asked questions",
    "list me some papers",
    "synonym for",
    "asp.net",
    "hows this sound",
    "blender",
    " seo ",
]

career_words = [
    "my career",
    "leadership",
    "management",
    "my resume",
    "mentorship",
    "i implemented",
    "work experience",
    "my training",
    "my resilience",
    "i am resilient",
    "hard work",
    "technical ",
    "organization",
    "my work",
    "my boss",
    " planning ",
    "i am professional",
    "i am ambitious",
    "i work",
    "at work",
    "i got a job",
]

creative_words = [
    "creativ",
    "stereotype",
    " my art ",
    "i appreciate",
    "i find it frustrating",
    "my personality",
    "assumptions about me",
    "assume about me",
    "my perception",
    "i perceive ",
    "feelings",
    "i feel",
    "photography",
    "sketching",
    " painting",
    "my emotions",
    "emotional",
    "my friend",
    "my mental",
    "i create",
    "i contemplate",
    "i struggle",
    "my struggles",
    "my sketch",
    "mental health",
    "i worry",
    "my identity",
    "i explore",
    "i sometimes",
    "i care",
    "i remember",
    "journaling",
    "i celebrate",
    "breathing",
    "compassion",
    "encouragement",
    "praise",
    "my habit",
    "motivation",
    "stress",
    "my hobbies",
    "mind",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-token",
        type=str,
        default="",
        help="Huggingface token that grants access to Wildchat dataset",
    )
    args = parser.parse_args()
    if os.path.isfile("data/wildchat_career_rewritten.csv"):
        print("Using rewritten data")
        df_career = pd.read_csv(
            "data/wildchat_career_rewritten.csv", skiprows=4, skipfooter=1
        )[["conversations", "replacement"]]
        df_creative = pd.read_csv(
            "data/wildchat_creative_rewritten.csv", skiprows=4, skipfooter=1
        )[["conversations", "replacement"]]
        df_career["conversations"] = df_career["replacement"].fillna(
            df_career["conversations"]
        )
        df_creative["conversations"] = df_creative["replacement"].fillna(
            df_creative["conversations"]
        )
    elif os.path.isfile("data/wildchat_career.csv"):
        print("Using saved data")
        df_career = pd.read_csv("data/wildchat_career.csv")
        df_creative = pd.read_csv("data/wildchat_creative.csv")

    else:
        print("Filtering wildchat")
        if args.token:
            login(args.token)

        # wildchat = (
        #     load_dataset("allenai/WildChat-1M")["train"]
        #     .filter(lambda example: example["language"] == "English")
        #     .shuffle(seed=42)
        # )
        # col = "conversation"

        wildchat = load_dataset("liweijiang/infinite-chats-taxonomy")[
            "train"
        ].shuffle(seed=42)
        col = "messages"

        career = []
        creative = []
        for i in tqdm(range(len(wildchat))):
            # text = " ".join([t["content"].lower() for t in wildchat[i][col][:2]])
            text = wildchat[i][col][0]["content"].lower()
            if any([e in text for e in exclude]) or text.split()[0] in [
                "write",
                "design",
                "role",
            ]:
                continue
            if len(career) < 100 and any([c in text for c in career_words]):
                career.append(wildchat[i][col][:10])
            elif len(creative) < 100 and any(
                [c in text for c in creative_words]
            ):
                creative.append(wildchat[i][col][:10])
            # elif any([c in text for c in creative_words]):
            #     print(wildchat[i][col])
            if i % 1000 == 0:
                print("Career: ", len(career), ", Creative: ", len(creative))

            if len(career) == 100 and len(creative) == 100:
                break

        df_career = pd.DataFrame({"conversations": career})
        df_career.to_csv("data/wildchat_career.csv")
        df_creative = pd.DataFrame({"conversations": creative})
        df_creative.to_csv("data/wildchat_creative.csv")

    if not os.path.isfile("data/wildchat_random.csv"):
        wildchat = load_dataset("liweijiang/infinite-chats-taxonomy")[
            "train"
        ].shuffle(seed=42)
        col = "messages"
        random = []
        career = [eval(c) for c in df_career["conversations"].tolist()]
        creative = [eval(c) for c in df_creative["conversations"].tolist()]
        for i in tqdm(range(len(wildchat))):
            text = wildchat[i][col][0]["content"].lower()
            if any([e in text for e in exclude]) or text.split()[0] in [
                "write",
                "design",
                "role",
            ]:
                continue
            if (
                wildchat[i][col][:10] in career
                or wildchat[i][col][:10] in creative
                or wildchat[i][col] in career
                or wildchat[i][col] in creative
            ):
                continue

            random.append(wildchat[i][col][:10])

            if len(random) == 100:
                break

        df_random = pd.DataFrame({"conversations": random})
        if type(df_random["conversations"].tolist()[0]) == str:
            df_random["conversations"] = df_random["conversations"].apply(
                ast.literal_eval
            )
        df_random.to_csv("data/wildchat_random.csv")
        with open("data/wildchat_random.pkl", "wb") as outfile:
            pickle.dump(df_random["conversations"].tolist(), outfile)

    if type(df_career["conversations"].tolist()[0]) == str:
        df_career["conversations"] = df_career["conversations"].apply(
            ast.literal_eval
        )
        df_creative["conversations"] = df_creative["conversations"].apply(
            ast.literal_eval
        )
    career = df_career["conversations"].tolist()
    creative = df_creative["conversations"].tolist()

    with open("data/wildchat_career.pkl", "wb") as outfile:
        pickle.dump(career, outfile)
    with open("data/wildchat_creative.pkl", "wb") as outfile:
        pickle.dump(creative, outfile)
