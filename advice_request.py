import argparse
import json
import os
from openai import OpenAI
import pandas as pd
import numpy as np
from tqdm import tqdm
import spacy

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print AITA posts and corresponding rewrites",
    )
    args = parser.parse_args()

    prompt = """Please rewrite the following request for advice such that it no longer uses framing specific to the AITA subreddit. This includes replacing abbreviations like 'AITA' or 'WIBTA', removing any "TLDR", "Update:" or "EDIT" sections, removing any references to comments or responses, and removing explicit declarations of gender and age in brackets like '(F25)'. Do not start with 'Request:', 'Request for advice:' or 'Advice request:'. Otherwise preserve the content of the advice request. Request: {text}"""
    nlp = spacy.load("en_core_web_sm")
    df = pd.read_csv("data/AITA-NTA-FLIP.csv")
    texts = []
    for i in range(len(df)):
        for prompt_type in ["original_post", "flipped_story"]:
            if df.iloc[i][prompt_type] == "ERROR":
                continue
            texts.append(df.iloc[i][prompt_type].strip().split("STORY: ")[-1])

    if os.path.isfile("batch_output_final.jsonl"):
        with open("batch_output_final.jsonl", "r") as json_file:
            json_list = list(json_file)

        output = [json.loads(f) for f in json_list]
        output = {
            int(o["custom_id"].split("-")[1]): o["response"]["body"][
                "choices"
            ][0]["message"]["content"]
            for o in output
        }
        overlap = []
        for i in tqdm(range(len(texts))):
            if args.verbose:

                print(texts[i])
                print(
                    "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
                )
                print(output[i])
                print(
                    "-----------------------------------------------------------------------------"
                )
            text_tokens = set([t.text.lower() for t in nlp(texts[i])])
            advice_tokens = set([t.text.lower() for t in nlp(output[i])])

            overlap.append(
                len(text_tokens.intersection(advice_tokens))
                / len(text_tokens)
                * 100
            )

        print(np.mean(overlap))

        idx = 0
        for i in range(len(df)):
            for prompt_type in ["original_post", "flipped_story"]:
                if df.iloc[i][prompt_type] == "ERROR":
                    continue
                df.loc[i, prompt_type] = output[idx]
                idx += 1
        df = df.drop(columns="Unnamed: 0")
        df.to_pickle("data/aita-nta-flip_rewritten.gz")
        df["len_original"] = df["original_post"].str.len()
        df["len_flipped"] = df["flipped_story"].str.len()
        df["len_diff"] = abs(df["len_original"] - df["len_flipped"])
        df = df.sort_values(by="len_diff").iloc[:1000]
        df = df.reset_index(drop=True)
        df.to_pickle("data/aita-nta-flip_rewritten_2000.gz")

    else:
        lines = [
            {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-5-mini",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant.",
                        },
                        {"role": "user", "content": prompt.format(text=text)},
                    ],
                },
            }
            for i, text in enumerate(texts)
        ]
        with open("batch_input_final.jsonl", "w") as f:
            for item in lines:
                f.write(json.dumps(item) + "\n")
        batch_input_file = client.files.create(
            file=open("batch_input_final.jsonl", "rb"), purpose="batch"
        )

        print(batch_input_file)
        batch_input_file_id = batch_input_file.id
        batch = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "rewrite AITA posts, full dataset"},
        )
        print(batch)
