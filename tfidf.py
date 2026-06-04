import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
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
    parser.add_argument(
        "--long", action="store_true", help="Whether to have long replies"
    )
    args = parser.parse_args()

    if args.long:
        df = pd.read_pickle(
            f"results/{args.model.split('/')[-1]}_long_results_liwc.pkl"
        )
        target_cols = [
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
    else:
        df = pd.read_pickle(f"results/{args.model.split('/')[-1]}_results.pkl")
        df_personas = pd.read_pickle("data/base_data_persona.gz")

        original = df[df["prompt_type"] == "original_post"][
            [
                "persona_type",
                "persona_id",
                "prompt_id",
                "model_answer_constrained",
            ]
        ].rename(
            columns={
                "model_answer_constrained": "model_answer_constrained_original_post",
            }
        )
        print("got original")

        flipped = df[df["prompt_type"] == "flipped_story"][
            [
                "persona_type",
                "persona_id",
                "prompt_id",
                "model_answer_constrained",
            ]
        ].rename(
            columns={
                "model_answer_constrained": "model_answer_constrained_flipped_story",
            }
        )
        print("got flipped")

        df = original.merge(
            flipped,
            on=["persona_type", "persona_id", "prompt_id"],
            how="outer",
        )

        print("got result")

        df = df.loc[(df["model_answer_constrained_original_post"] == "No")]
        df["sycophancy"] = df["model_answer_constrained_flipped_story"] == "No"
        target_cols = ["sycophancy"]
        df = df.merge(df_personas, on=["persona_id", "persona_type"])
        del df_personas

    vectorizer = TfidfVectorizer(min_df=10, max_df=0.7)
    X = vectorizer.fit_transform(tqdm(df["persona_text"].tolist()))
    feature_names = vectorizer.get_feature_names_out()

    for target_col in target_cols:
        best_model = Ridge().fit(
            X,
            df[target_col],
        )
        coefs = best_model.coef_
        score = best_model.score(
            X,
            df[target_col],
        )

        print(f"Score: {score}")

        # ── 4. Filter to non-zero (selected) features ────────────────────────────────
        mask = coefs != 0
        sel_names = np.array(feature_names)[mask]
        sel_coefs = coefs[mask]

        # Sort by absolute value descending
        order = np.argsort(np.abs(sel_coefs))[::-1]
        sel_names = sel_names[order]
        sel_coefs = sel_coefs[order]

        # ── 5. Plot — 8-column layout ─────────────────────────────────────────────────
        POS_COLOR = "#2E86AB"  # blue  – positive coefficients
        NEG_COLOR = "#E84855"  # red   – negative coefficients
        TEXT_COLOR = "#0F1117"
        GRID_COLOR = "#D3D3D3"
        # BG_COLOR = "#E8EAF0"

        TOP_N = 50
        top_names = sel_names[:TOP_N]
        top_coefs = sel_coefs[:TOP_N]

        fig, ax = plt.subplots(figsize=(16, 6))

        x_pos = np.arange(len(top_names))
        colors = [POS_COLOR if c > 0 else NEG_COLOR for c in top_coefs]

        bars = ax.bar(
            x_pos,
            top_coefs,
            color=colors,
            width=0.65,
            edgecolor="none",
            zorder=3,
        )

        # Value labels above/below each bar
        y_range = max(abs(top_coefs))
        for bar, val in zip(bars, top_coefs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.03 * y_range * np.sign(val),
                f"{val:+.2f}",
                ha="center",
                va="bottom" if val > 0 else "top",
                fontsize=7.5,
                color=TEXT_COLOR,
                alpha=0.85,
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            top_names, rotation=35, ha="right", fontsize=9, color=TEXT_COLOR
        )
        ax.tick_params(axis="y", colors=TEXT_COLOR, labelsize=8)
        ax.axhline(0, color=TEXT_COLOR, linewidth=0.8, alpha=0.4, zorder=2)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6, zorder=1)
        ax.set_ylabel(
            "Standardised Coefficient",
            color=TEXT_COLOR,
            fontsize=10,
            labelpad=8,
        )
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

        # Title & subtitle
        fig.text(
            0.5,
            0.98,
            "Top 50 Features by Coefficient Magnitude",
            fontsize=13,
            fontweight="bold",
            color=TEXT_COLOR,
            va="top",
            ha="center",
        )
        fig.text(
            0.5,
            0.945,
            f"$R^2$ = {score:.3f}",
            fontsize=8.5,
            color=TEXT_COLOR,
            alpha=0.6,
            va="top",
            ha="center",
        )

        # Legend
        pos_patch = mpatches.Patch(color=POS_COLOR, label="Positive effect")
        neg_patch = mpatches.Patch(color=NEG_COLOR, label="Negative effect")
        ax.legend(
            handles=[pos_patch, neg_patch],
            frameon=False,
            fontsize=9,
            labelcolor=TEXT_COLOR,
            loc="upper right",
        )

        plt.tight_layout(rect=[0, 0, 1, 0.93])

        plt.savefig(
            f"figures/{args.model.split('/')[-1]}_{target_col}_tfidf.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.show()
        print(
            f"Plot saved to figures/{args.model.split('/')[-1]}_{target_col}_tfidf.png"
        )
