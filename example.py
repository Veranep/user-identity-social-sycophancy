import argparse
from datasets import load_dataset
from huggingface_hub import login
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

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
        "-token",
        type=str,
        default="",
        help="Huggingface token that grants access to Llama model",
    )
    args = parser.parse_args()
    if args.token:
        login(args.token)
    np.random.seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if "gemma" in args.model:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    model = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if not model.tokenizer.pad_token_id:
        model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
    df = pd.read_csv("data/AITA-NTA-FLIP.csv")
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
    data = {
        "persona_type": [],
        "persona_id": [],
        "persona_text": [],
        "prompt_type": [],
        "prompt_id": [],
        "prompt_text": [],
    }
    for i in tqdm(range(5)):
        for j in tqdm(range(5)):
            for prompt_type in ["original_prompt", "flipped_story"]:
                if df.iloc[i][prompt_type] == "ERROR":
                    continue
                for persona_type in ["hobbies", "life_story"]:
                    data["persona_type"].append(persona_type)
                    data["persona_id"].append(df_personas.iloc[j]["uuid"])
                    data["persona_text"].append(
                        df_personas.iloc[j][persona_type]
                    )
                    data["prompt_type"].append(prompt_type)
                    data["prompt_id"].append(df.iloc[i]["id"])
                    data["prompt_text"].append(
                        df.iloc[i][prompt_type].strip().split("STORY: ")[1]
                    )
                data["persona_type"].append("both")
                data["persona_id"].append(df_personas.iloc[j]["uuid"])
                data["persona_text"].append(
                    df_personas.iloc[j]["life_story"]
                    + " "
                    + df_personas.iloc[j]["hobbies"]
                )
                data["prompt_type"].append(prompt_type)
                data["prompt_id"].append(df.iloc[i]["id"])
                data["prompt_text"].append(
                    df.iloc[i][prompt_type].strip().split("STORY: ")[1]
                )
                data["persona_type"].append(None)
                data["persona_id"].append(None)
                data["persona_text"].append(None)
                data["prompt_type"].append(prompt_type)
                data["prompt_id"].append(df.iloc[i]["id"])
                data["prompt_text"].append(
                    df.iloc[i][prompt_type].strip().split("STORY: ")[1]
                )

    df_data = pd.DataFrame(data)
    print(df_data)
    current_convos = [
        [{"role": "user", "content": persona}]
        for persona in df_data["persona_text"].tolist()
        if persona
    ]
    current_convos = [
        convo[0]["generated_text"]
        for convo in tqdm(
            model(
                current_convos,
                batch_size=batch_size,
                do_sample=False,
                max_new_tokens=500,
            ),
            total=len(current_convos),
        )
    ]
    final_convos = []
    final_convos_constrained = []
    for i, convo in enumerate(current_convos):
        final_convos.append(
            convo
            + [{"role": "user", "content": df_data.iloc[i]["prompt_text"]}]
        )
        final_convos_constrained.append(
            convo
            + [
                {
                    "role": "user",
                    "content": df_data.iloc[i]["prompt_text"]
                    + "\nOutput only YTA or NTA.",
                }
            ]
        )
    final_convos += [
        [{"role": "user", "content": prompt_text}]
        for prompt_text in df_data.loc[df_data["persona_type"].isna()][
            "prompt_text"
        ].tolist()
    ]

    final_convos_constrained += [
        [
            {
                "role": "user",
                "content": prompt_text + "\nOutput only YTA or NTA.",
            }
        ]
        for prompt_text in df_data.loc[df_data["persona_type"].isna()][
            "prompt_text"
        ].tolist()
    ]

    df_data["model_answer"] = [
        answer[0]["generated_text"][-1]["content"]
        for answer in tqdm(
            model(
                final_convos,
                batch_size=32,
                do_sample=False,
                max_new_tokens=500,
            ),
            total=len(final_convos),
        )
    ]

    df_data["model_answer_constrained"] = [
        answer[0]["generated_text"][-1]["content"]
        for answer in tqdm(
            model(
                final_convos_constrained,
                batch_size=32,
                do_sample=False,
                max_new_tokens=5,
            ),
            total=len(final_convos_constrained),
        )
    ]

    df_data.to_pickle("results/example_results.pkl")
