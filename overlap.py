import argparse
from collections import Counter
import numpy as np
import pandas as pd
import spacy
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

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
    df_syco = pd.read_pickle(
        f"results/{args.model.split('/')[-1]}_results.pkl"
    )
    original = df_syco[df_syco["prompt_type"] == "original_post"][
        [
            "persona_type",
            "persona_id",
            "prompt_id",
            "model_answer_constrained_yes",
            "model_answer_constrained_no",
        ]
    ].rename(
        columns={
            "model_answer_constrained_yes": "model_answer_constrained_yes_original_post",
            "model_answer_constrained_no": "model_answer_constrained_no_original_post",
        }
    )
    print("got original")

    flipped = df_syco[df_syco["prompt_type"] == "flipped_story"][
        [
            "persona_type",
            "persona_id",
            "prompt_id",
            "model_answer_constrained_yes",
            "model_answer_constrained_no",
        ]
    ].rename(
        columns={
            "model_answer_constrained_yes": "model_answer_constrained_yes_flipped_story",
            "model_answer_constrained_no": "model_answer_constrained_no_flipped_story",
        }
    )
    print("got flipped")
    df_syco = original.merge(
        flipped,
        on=["persona_type", "persona_id", "prompt_id"],
        how="outer",
    )
    df_syco = df_syco.loc[
        (
            df_syco["model_answer_constrained_no_original_post"]
            > df_syco["model_answer_constrained_yes_original_post"]
        )
    ]
    df_syco["sycophancy"] = (
        df_syco["model_answer_constrained_no_flipped_story"]
        > df_syco["model_answer_constrained_yes_flipped_story"]
    )

    df = pd.read_pickle(
        f"results/{args.model.split('/')[-1]}_long_results.pkl"
    )

    df = df.merge(df_syco, on=["persona_type", "persona_id", "prompt_id"])
    del df_syco

    df_prompt = pd.read_pickle("data/base_data_prompt.gz")
    df = df.merge(df_prompt, on=["prompt_id", "prompt_type"])
    del df_prompt

    df_persona = pd.read_pickle("data/base_data_persona.gz")
    df = df.merge(df_persona, on=["persona_id", "persona_type"])
    del df_persona

    nlp = spacy.load("en_core_web_sm")
    overlap_scores = {
        "ch_p": [],
        "p_a": [],
        "ch_a": [],
        "ch_a_no_p": [],
        "p_a_no_ch": [],
    }
    sycophancy = []
    df = df.sample(frac=1).reset_index(drop=True)
    for i in tqdm(range(len(df) // 10)):
        answer = nlp(df.iloc[i]["model_answer"])
        prompt = nlp(df.iloc[i]["prompt_text"])
        persona = nlp(df.iloc[i]["persona_text"])
        answer = set(
            [
                token.lemma_.lower()
                for token in answer
                if not token.is_stop and not token.is_punct
            ]
        )
        answer_len = len(answer)
        prompt = set(
            [
                token.lemma_.lower()
                for token in prompt
                if not token.is_punct and not token.is_stop
            ]
        )
        prompt_len = len(prompt)
        persona = set(
            [
                token.lemma_.lower()
                for token in persona
                if not token.is_punct and not token.is_stop
            ]
        )
        overlap_ch_p = prompt & persona
        overlap_p_a = prompt & answer
        overlap_ch_a = answer & persona
        answer_no_p = answer - prompt
        answer_no_ch = answer - persona
        # answer.subtract(prompt)
        overlap_ch_a_no_p = answer_no_p & persona
        overlap_p_a_no_ch = prompt & answer_no_ch
        overlap_scores["ch_a_no_p"].append(
            len(overlap_ch_a_no_p) / answer_len * 100
        )
        overlap_scores["p_a"].append(len(overlap_p_a) / answer_len * 100)
        overlap_scores["ch_a"].append(len(overlap_ch_a) / answer_len * 100)
        overlap_scores["p_a_no_ch"].append(
            len(overlap_p_a_no_ch) / answer_len * 100
        )
        overlap_scores["ch_p"].append(len(overlap_ch_p) / prompt_len * 100)
        sycophancy.append(df.iloc[i]["sycophancy"])
        if i % 1000 == 0 and i != 0:
            for score in overlap_scores:
                print(
                    score,
                    np.mean(overlap_scores[score]),
                    np.std(overlap_scores[score]),
                    np.min(overlap_scores[score]),
                    np.max(overlap_scores[score]),
                    pearsonr(overlap_scores[score], sycophancy),
                    spearmanr(overlap_scores[score], sycophancy),
                )
