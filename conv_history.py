from datasets import load_dataset
import json
import os
from openai import OpenAI
import pandas as pd
from tqdm import tqdm

# client = OpenAI(
#     # This is the default and can be omitted
#     api_key=os.environ.get("OPENAI_API_KEY"),
# )

if __name__ == "__main__":
    prompt = """Given the following information about a user, please generate a realistic conversation between that user and an LLM, in which the user shares this information. Use the following format:
[{{"role": "user", "content": "user message"}}, {{"role": "assistant": "content": "model reply"}}]
Make sure the last turn is by the assistant and that this turn does not end with a question to the user.
User information:
{text}"""
    df_personas = load_dataset("Salesforce/SCOPE-Persona")["train"].to_pandas()
    df_personas["hobbies"] = df_personas[
        "behavioral_patterns_and_preferences"
    ].apply(lambda x: dict(x)["What are your main hobbies and interests?"])
    df_personas["life_story"] = df_personas[
        "personal_identity_and_life_narratives"
    ].apply(
        lambda x: dict(x)[
            "To start, I would like to begin with a big question: tell me the story of your life. Start from the beginning–from your childhood, to education, and to any major life events you may have had."
        ]
    )
    df_personas["family"] = df_personas[
        "personal_identity_and_life_narratives"
    ].apply(
        lambda x: dict(x)[
            "Tell me about people and relations in your life. Are there people outside and within of your family who are important to you?"
        ]
    )
    df_personas["setback"] = df_personas[
        "personal_identity_and_life_narratives"
    ].apply(
        lambda x: dict(x)[
            "Can you describe a time when you faced setbacks and how you overcame them?"
        ]
    )
    texts = []
    for i in tqdm(range(5)):
        for persona_type in [
            "hobbies",
            "life_story",
            "family",
            "setback",
        ]:
            texts.append(df_personas.iloc[i][persona_type])
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
    with open("batch_input_ch.jsonl", "w") as f:
        for item in lines:
            f.write(json.dumps(item) + "\n")
    # batch_input_file = client.files.create(
    #     file=open("batch_input_ch.jsonl", "rb"), purpose="batch"
    # )

    # print(batch_input_file)
    # batch_input_file_id = batch_input_file.id
    # batch = client.batches.create(
    #     input_file_id=batch_input_file_id,
    #     endpoint="/v1/chat/completions",
    #     completion_window="24h",
    #     metadata={"description": "short eval job"},
    # )
    # print(batch)
