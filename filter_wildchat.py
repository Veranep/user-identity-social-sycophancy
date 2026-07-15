import argparse
from datasets import load_dataset
from huggingface_hub import login
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

    wildchat = load_dataset("allenai/WildChat-1M")["train"].filter(
        lambda example: example["language"] == "English"
    )

    career = []
    creative = []
    for i in tqdm(range(len(wildchat))):
        text = " ".join(
            [t["content"].lower() for t in wildchat[i]["conversation"]]
        )
        if len(career) < 100 and (
            "career" in text or "leadership" in text or "management" in text
        ):
            career.append(wildchat[i]["conversation"])

        if len(creative) < 100 and (
            "creativity" in text or "stereotype" in text or "feelings" in text
        ):
            creative.append(wildchat[i]["conversation"])

        if len(career) == 100 and len(creative) == 100:
            break
    print(career[:5])
    print(creative[:5])
