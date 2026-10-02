import argparse
from datasets import load_dataset
from huggingface_hub import login
import json
import os
from openai import OpenAI
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import pipeline
import spacy

from get_model_responses import get_persona_convo

# client = OpenAI(
#     # This is the default and can be omitted
#     api_key=os.environ.get("OPENAI_API_KEY"),
#     # base_url="https://eu.api.openai.com/v1",
# )


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
        "--verbose",
        action="store_true",
        help="Print persona identity descriptions and corresponding conversation histories",
    )
    args = parser.parse_args()
    prompt = """Given the following information about a user, please generate a realistic conversation between that user and an LLM, in which the user shares this information. Use the following format:
[{{"role": "user", "content": "user message"}}, {{"role": "assistant": "content": "model reply"}}]
Make sure the last turn is by the assistant and that this turn does not end with a question to the user.
User information:
{text}"""
    n = 200
    nlp = spacy.load("en_core_web_sm")
    if not os.path.isfile(f"data/personas_{n}.gz"):
        df_personas = (
            load_dataset(
                "Salesforce/SCOPE-Persona",
                revision="154873d548cd2fd8055ccb38fe80117638071dce",
            )["train"]
            .shuffle(seed=21)
            .select(list(range(500)))
            .to_pandas()
        )
        df_personas["hobbies"] = df_personas[
            "behavioral_patterns_and_preferences"
        ].apply(lambda x: dict(x)["What are your main hobbies and interests?"])
        df_personas["motivation"] = df_personas[
            "behavioral_patterns_and_preferences"
        ].apply(
            lambda x: dict(x)[
                "How do you stay motivated when things don’t go as planned?"
            ]
        )
        df_personas["recognition"] = df_personas[
            "behavioral_patterns_and_preferences"
        ].apply(
            lambda x: dict(x)[
                "How do you feel when others recognize your achievements?"
            ]
        )
        df_personas["life_story"] = df_personas[
            "personal_identity_and_life_narratives"
        ].apply(
            lambda x: dict(x)[
                "To start, I would like to begin with a big question: tell me the story of your life. Start from the beginning–from your childhood, to education, and to any major life events you may have had."
            ]
        )
        df_personas["crossroads"] = df_personas[
            "personal_identity_and_life_narratives"
        ].apply(
            lambda x: dict(x)[
                "Some people tell us that they’ve reached a crossroads at some points in their life where multiple paths were available, and their choice then made a significant difference in defining who they are."
            ]
        )
        df_personas["family"] = df_personas[
            "personal_identity_and_life_narratives"
        ].apply(
            lambda x: dict(x)[
                "Tell me about people and relations in your life. Are there people outside and within of your family who are important to you?"
            ]
        )
        df_personas["influence"] = df_personas[
            "personal_identity_and_life_narratives"
        ].apply(
            lambda x: dict(x)[
                "Who has had the biggest influence on your personal journey?"
            ]
        )
        df_personas["setback"] = df_personas[
            "personal_identity_and_life_narratives"
        ].apply(
            lambda x: dict(x)[
                "Can you describe a time when you faced setbacks and how you overcame them?"
            ]
        )
        df_personas["politics"] = df_personas[
            "personal_identity_and_life_narratives"
        ].apply(
            lambda x: dict(x)[
                "How do you think about your political beliefs, civic participation, or involvement in your community?"
            ]
        )
        df_personas["assumptions"] = df_personas[
            "personal_identity_and_life_narratives"
        ].apply(
            lambda x: dict(x)[
                "Have you ever felt that people’s assumptions about you didn’t match who you are? If so how?"
            ]
        )
        df_personas["age"] = df_personas["demographic_information"].apply(
            lambda x: dict(x)["Select Your Age"]
        )
        df_personas["grand_children"] = pd.DataFrame(
            {
                col: df_personas[col].str.contains("grandchild")
                for col in [
                    "motivation",
                    "recognition",
                    "life_story",
                    "crossroads",
                    "family",
                    "influence",
                    "setback",
                    "politics",
                    "assumptions",
                ]
            }
        ).any(axis=1)
        df_personas["gender_mention"] = pd.DataFrame(
            {
                col: df_personas[col].str.contains(
                    r"\b(?!([Hh]uman|German)\b)[a-zA-Z]*man\b"
                )
                for col in [
                    "motivation",
                    "recognition",
                    "life_story",
                    "crossroads",
                    "family",
                    "influence",
                    "setback",
                    "politics",
                    "assumptions",
                ]
            }
        ).any(axis=1)
        df_personas = df_personas.loc[
            ~df_personas["grand_children"]
        ].reset_index(drop=True)
        df_personas = df_personas.loc[
            ~df_personas["gender_mention"]
        ].reset_index(drop=True)
        df_personas = df_personas.loc[
            ~df_personas["age"].isin(
                ["Less than 20", "60 - 64", "65 or older", "70 or older"]
            )
        ].reset_index(drop=True)
        df_personas = df_personas.iloc[:n]
        df_personas = df_personas.drop(
            columns=["grand_children", "gender_mention"]
        )
        df_personas.to_pickle(f"data/personas_{n}.gz")

    else:
        df_personas = pd.read_pickle(f"data/personas_{n}.gz")

    print(df_personas)

    persona_types = [
        "hobbies",
        "motivation",
        "recognition",
        "life_story",
        "crossroads",
        "family",
        "influence",
        "setback",
        "politics",
        "assumptions",
    ]

    texts = []
    for i in tqdm(range(n)):
        for persona_type in persona_types:
            texts.append(df_personas.iloc[i][persona_type])

    if "gpt" in args.model:
        if os.path.isfile("batch_output_ch_final.jsonl"):
            with open("batch_output_ch_final.jsonl", "r") as json_file:
                json_list = list(json_file)

            output = [json.loads(f) for f in json_list]
            output = {
                int(o["custom_id"].split("-")[1]): o["response"]["body"][
                    "choices"
                ][0]["message"]["content"]
                for o in output
            }
            overlap = []
            for i in tqdm(range(len(persona_types) * n)):
                if args.verbose:
                    print(texts[i])
                    print(
                        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
                    )
                    print(output[i])
                    print(
                        "-----------------------------------------------------------------------------"
                    )
                persona_type = persona_types[i % len(persona_types)]
                idx = i // len(persona_types)
                df_personas.loc[idx, persona_type] = output[i]

                conv_history = " ".join(
                    [
                        turn["content"]
                        for turn in get_persona_convo(output[i])
                        if turn["role"] == "user"
                    ]
                )

                text_tokens = set([t.text.lower() for t in nlp(texts[i])])
                conv_history_tokens = set(
                    [t.text.lower() for t in nlp(conv_history)]
                )

                overlap.append(
                    len(text_tokens.intersection(conv_history_tokens))
                    / len(text_tokens)
                    * 100
                )

            if not os.path.isfile(f"data/personas_{n}_ch.gz"):
                df_personas.to_pickle(f"data/personas_{n}_ch.gz")

            print(np.mean(overlap))

        # batch = client.batches.retrieve('batch_...')
        # file_response = client.files.content('file-...')
        # with open("batch_output_ch_final.jsonl", "w") as f:
        #    f.write(file_response.text)

        else:
            pass
            # lines = [
            #     {
            #         "custom_id": f"request-{i}",
            #         "method": "POST",
            #         "url": "/v1/chat/completions",
            #         "body": {
            #             "model": "gpt-5-mini",
            #             "messages": [
            #                 {
            #                     "role": "system",
            #                     "content": "You are a helpful assistant.",
            #                 },
            #                 {"role": "user", "content": prompt.format(text=text)},
            #             ],
            #         },
            #     }
            #     for i, text in enumerate(texts)
            # ]
            # with open("batch_input_ch_final.jsonl", "w") as f:
            #     for item in lines:
            #         f.write(json.dumps(item) + "\n")
            # batch_input_file = client.files.create(
            #     file=open("batch_input_ch_final.jsonl", "rb"), purpose="batch"
            # )

            # print(batch_input_file)
            # batch_input_file_id = batch_input_file.id
            # batch = client.batches.create(
            #     input_file_id=batch_input_file_id,
            #     endpoint="/v1/chat/completions",
            #     completion_window="24h",
            #     metadata={
            #         "description": f"write conversation history for {n} personas"
            #     },
            # )
            # print(batch)
    elif not os.path.isfile(f"data/personas_{n}_{model}_ch.gz"):
        if args.token:
            login(args.token)
        model = pipeline(
            "text-generation",
            model=args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        if not model.tokenizer.pad_token_id:
            model.tokenizer.pad_token_id = model.tokenizer.eos_token_id

        texts = [
            model.tokenizer.apply_chat_template(
                prompt.format(text=text),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for text in texts
        ]
        output = [
            answer[0]["generated_text"]
            for answer in tqdm(
                model(
                    ListDataset(texts),
                    batch_size=32,
                    do_sample=False,
                    return_full_text=False,
                    max_new_tokens=500,
                ),
                total=len(texts),
            )
        ]
        overlap = []
        for i in tqdm(range(len(persona_types) * n)):
            if args.verbose:
                print(texts[i])
                print(
                    "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
                )
                print(output[i])
                print(
                    "-----------------------------------------------------------------------------"
                )
            persona_type = persona_types[i % len(persona_types)]
            idx = i // len(persona_types)
            df_personas.loc[idx, persona_type] = output[i]

        df_personas.to_pickle(f"data/personas_{n}_{model}_ch.gz")

    else:
        df_personas = pd.read_pickle(f"data/personas_{n}_{model}_ch.gz")
        for i in tqdm(range(len(persona_types) * n)):
            conv_history = " ".join(
                [
                    turn["content"]
                    for turn in get_persona_convo(output[i])
                    if turn["role"] == "user"
                ]
            )

            text_tokens = set([t.text.lower() for t in nlp(texts[i])])
            conv_history_tokens = set(
                [t.text.lower() for t in nlp(conv_history)]
            )

            overlap.append(
                len(text_tokens.intersection(conv_history_tokens))
                / len(text_tokens)
                * 100
            )

        print(np.mean(overlap))
