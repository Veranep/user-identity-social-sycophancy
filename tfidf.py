#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV


# In[2]:


model = "gemma-3-12b-it"


# In[3]:


df = pd.read_pickle(f"results/{model.split('/')[-1]}_results.pkl")

original = df[df["prompt_type"] == "original_post"][
        [
            "persona_type",
            "persona_id",
            "persona_text",
            "prompt_id",
            "prompt_text",
            "model_answer_constrained"
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
        "model_answer_constrained"
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

print("got result")


# In[4]:


df = df.loc[
        (
            df["model_answer_constrained_original_post"] == "No"
        )
]
df["flipped_is_na"] = (
        df["model_answer_constrained_flipped_story"] == "No"
)


# In[ ]:


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["persona_text"])
feature_names = vectorizer.get_feature_names_out()


# In[ ]:


l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1]  # α candidates
en_model = ElasticNetCV(
    l1_ratio=l1_ratios,
    cv=5,
    max_iter=10000,
    random_state=42,
)

best_model = en_model.fit(
    X,
    df["flipped_is_na"],
)

best_alpha = en_model.alpha_
best_l1 = en_model.l1_ratio_
coefs = en_model.coef_
score = best_model.score(
    X,
    df["flipped_is_na"],
)

print(f"Best α (alpha) : {best_alpha:.4f}")
print(
    f"Best λ (l1_ratio): {best_l1:.2f}  "
    f"({'pure Lasso' if best_l1==1 else 'pure Ridge' if best_l1==0 else 'Elastic Net'})"
)
print(f"Non-zero coefficients: {np.sum(coefs != 0)} / {len(coefs)}")
print(f"Score: {score}")

# ── 4. Filter to non-zero (selected) features ────────────────────────────────
mask = coefs != 0
sel_names = np.array(feature_names)[mask]
sel_coefs = coefs[mask]

if len(sel_names) == 0:
    continue

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
# fig.patch.set_facecolor(BG_COLOR)
# ax.set_facecolor(BG_COLOR)

x_pos = np.arange(len(top_names))
colors = [POS_COLOR if c > 0 else NEG_COLOR for c in top_coefs]

bars = ax.bar(
    x_pos, top_coefs, color=colors, width=0.65, edgecolor="none", zorder=3
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
ax.set_xticklabels(top_names, rotation=35, ha="right", fontsize=9, color=TEXT_COLOR)
ax.tick_params(axis="y", colors=TEXT_COLOR, labelsize=8)
ax.axhline(0, color=TEXT_COLOR, linewidth=0.8, alpha=0.4, zorder=2)
ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6, zorder=1)
ax.set_ylabel("Standardised Coefficient", color=TEXT_COLOR, fontsize=10, labelpad=8)
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
    f"λ = {best_alpha:.4f}   |   α (l1_ratio) = {best_l1:.2f}   |   "
    f"{len(sel_coefs)} of {len(en_model.coef_)} features selected   |   "
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
    f"figures/{model}_sycophancy_tfidf.png",
    dpi=150,
    bbox_inches="tight",
    # facecolor=BG_COLOR,
)
plt.show()
print(f"Plot saved to figures/{model}_sycophancy_tfidf.png")

