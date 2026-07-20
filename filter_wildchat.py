import argparse
from datasets import load_dataset
from huggingface_hub import login
import pickle
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-token",
        type=str,
        default="",
        help="Huggingface token that grants access to Llama model",
    )
    args = parser.parse_args()
    if args.token:
        login(args.token)

    # wildchat = load_dataset("allenai/WildChat-1M")["train"].filter(
    #     lambda example: example["language"] == "English"
    # ).shuffle(seed=42)
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
        if (
            len(career) < 100
            and (
                "career" in text
                or "leadership" in text
                or "management" in text
            )
            and len(wildchat[i][col]) <= 10
        ):
            career.append(wildchat[i][col])

        if (
            len(creative) < 100
            and (
                "creativity" in text
                or "stereotype" in text
                or "artistic" in text
                or "appreciate" in text
                or "frustrating" in text
                or "personality" in text
                or "assumptions" in text
                or "perception" in text
                or "perceive " in text
                or "feelings" in text
                or "i feel" in text
                or "other people" in text
                or "someone" in text
            )
            and not (
                "write a" in text
                or "create me" in text
                or "generate for me" in text
                or "story" in text
                or "write me" in text
                or "python" in text
            )
            and len(wildchat[i][col]) <= 10
        ):
            creative.append(wildchat[i][col])

        if len(career) == 100 and len(creative) == 100:
            break
    with open("data/wildchat_career.pkl", "wb") as outfile:
        pickle.dump(career, outfile)
    with open("data/wildchat_creative.pkl", "wb") as outfile:
        pickle.dump(creative, outfile)
