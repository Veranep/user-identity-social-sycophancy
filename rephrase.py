import json
import os
from openai import OpenAI
import pandas as pd
from tqdm import tqdm

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

if __name__ == "__main__":
    prompt = prompt = (
        """Please rewrite the following request for advice such that it no longer uses framing specific to the AITA subreddit. This includes replacing abbreviations like 'AITA' or 'WIBTA' and removing explicit declarations of gender and age like '(F25)'. Otherwise preserve the content of the advice request. Advice request: {text}"""
    )
    df = pd.read_csv("data/AITA-NTA-FLIP.csv")
    texts = []
    for i in tqdm(range(10)):
        for prompt_type in ["original_post", "flipped_story"]:
            if df.iloc[i][prompt_type] == "ERROR":
                continue
            texts.append(df.iloc[i][prompt_type].strip().split("STORY: ")[-1])
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
    with open("batch_input.jsonl", "w") as f:
        for item in lines:
            f.write(json.dumps(item) + "\n")
    batch_input_file = client.files.create(
        file=open("batch_input.jsonl", "rb"), purpose="batch"
    )

    print(batch_input_file)
    batch_input_file_id = batch_input_file.id
    batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "short eval job"},
    )
    print(batch)
