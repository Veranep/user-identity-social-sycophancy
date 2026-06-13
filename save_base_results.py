import pandas as pd
from tqdm import tqdm

from get_model_responses import get_persona_convo

df = pd.read_pickle("data/aita-nta-flip_rewritten_2000.gz")
n = 200
df_personas = pd.read_pickle(f"data/personas_{n}_ch.gz")
persona_data = {
    "persona_type": [],
    "persona_id": [],
    "persona_text": [],
}
persona_data_turns = {
    "persona_type": [],
    "persona_id": [],
    "content": [],
    "role": [],
}
prompt_data = {
    "prompt_type": [],
    "prompt_id": [],
    "prompt_text": [],
}

for j in tqdm(range(len(df_personas))):
    p_id = df_personas.iloc[j]["uuid"]
    for persona_type in [
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
    ]:
        persona_data["persona_type"].append(persona_type)
        persona_data["persona_id"].append(p_id)
        persona_data["persona_text"].append(df_personas.iloc[j][persona_type])
        for turn in get_persona_convo(df_personas.iloc[j][persona_type]):
            persona_data_turns["persona_type"].append(persona_type)
            persona_data_turns["persona_id"].append(p_id)
            persona_data_turns["content"].append(turn["content"])
            persona_data_turns["role"].append(turn["role"])

persona_data["persona_type"].append(None)
persona_data["persona_id"].append(None)
persona_data["persona_text"].append("[]")

df_persona = pd.DataFrame(persona_data)
df_persona.to_pickle("data/base_data_persona.gz")
df_persona_turns = pd.DataFrame(persona_data_turns)
df_persona_turns.to_pickle("data/base_data_persona_turns.gz")


for i in tqdm(range(len(df))):
    for prompt_type in ["original_post", "flipped_story"]:
        prompt_data["prompt_type"].append(prompt_type)
        prompt_data["prompt_id"].append(df.iloc[i]["id"])
        prompt_data["prompt_text"].append(df.iloc[i][prompt_type])

df_prompt = pd.DataFrame(prompt_data)
df_prompt.to_pickle("data/base_data_prompt.gz")
