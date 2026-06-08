import pandas as pd

if __name__ == "__main__":
    df = pd.read_pickle("results/old/gemma-3-12b-it_results.pkl")
    liwc = pd.read_pickle("results/gemma-3-12b-it_long_results_liwc.pkl")
    original = df[df["prompt_type"] == "original_post"][
        [
            "persona_type",
            "persona_id",
            "persona_text",
            "prompt_id",
            "prompt_text",
            "model_answer_constrained",
        ]
    ].rename(
        columns={
            "prompt_text": "prompt_text_original",
            "model_answer_constrained": "model_answer_constrained_original_post",
        }
    )
    print("got original")

    flipped = df[df["prompt_type"] == "flipped_story"][
        [
            "persona_type",
            "persona_id",
            "persona_text",
            "prompt_id",
            "prompt_text",
            "model_answer_constrained",
        ]
    ].rename(
        columns={
            "prompt_text": "prompt_text_flipped",
            "model_answer_constrained": "model_answer_constrained_flipped_story",
        }
    )
    print("got flipped")
    df = original.merge(
        flipped,
        on=["persona_type", "persona_id", "persona_text", "prompt_id"],
        how="outer",
    )
    df = df.loc[(df["model_answer_constrained_original_post"] == "No")]
    df["sycophancy"] = df["model_answer_constrained_flipped_story"] == "No"
    df = (
        df.groupby(["persona_type", "persona_id", "persona_text"])[
            "sycophancy"
        ]
        .mean()
        .reset_index()
    )

    liwc = (
        liwc.drop(
            columns=["prompt_type", "prompt_text", "model_answer", "prompt_id"]
        )
        .groupby(["persona_type", "persona_id", "persona_text"])
        .mean()
        .reset_index()
    )
    df = pd.merge(df, liwc, on=["persona_type", "persona_id", "persona_text"])
    print(
        df[
            [
                "sycophancy",
                "liwc_emotion",
                "liwc_emo_pos",
                "liwc_emo_neg",
                "liwc_emo_anx",
                "liwc_emo_anger",
                "liwc_emo_sad",
                "liwc_swear",
                "liwc_Social",
                "liwc_socbehav",
                "liwc_prosocial",
                "liwc_cogproc",
                "liwc_Cognition",
                "liwc_Affect",
                "liwc_tone_pos",
                "liwc_tone_neg",
                "liwc_insight",
            ]
        ].corr()
    )
