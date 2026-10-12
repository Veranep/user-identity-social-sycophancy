import argparse
from collections import Counter
from marked_words import (
    get_marked_words_short,
    get_marked_words_short_advice,
)
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
from tqdm import tqdm
from wordcloud import WordCloud

from get_model_responses import get_persona_convo

all_models = [
    "Qwen3.6-35B-A3B",
    "gemma-3-12b-it",
    "gemma-3-27b-it",
    "Llama-3.1-8B-Instruct",
    "Qwen3.6-27B",
    "Olmo-3-7B-Instruct",
    "Qwen3-8B",
]

wc_models = [
    "gemma-3-12b-it",
    "Llama-3.1-8B-Instruct",
    "Llama-3.1-70B-Instruct",
    "gpt-5.6-luna",
    "gemini-3.5-flash-lite",
    "Qwen3.6-27B",
    "Olmo-3-7B-Instruct",
    "Qwen3-8B",
]


def extract_roles(convos, e_word, wc=False):
    roles = {"user": 0, "assistant": 0, "both_uf": 0, "both_af": 0}
    for convo in convos:
        if not wc:
            convo = get_persona_convo(convo)
        c_roles = {"user": 0, "assistant": 0}
        first = None
        for turn in convo:
            split_text = turn["content"].lower().split()
            if e_word in split_text:
                c_roles[turn["role"]] += 1
                if not first:
                    first = turn["role"]
        if c_roles["user"] and c_roles["assistant"]:
            if first == "user":
                roles["both_uf"] += 1
            else:
                roles["both_af"] += 1
        elif c_roles["user"]:
            roles["user"] += 1
        elif c_roles["assistant"]:
            roles["assistant"] += 1
    print(
        roles["user"],
        "\t",
        roles["assistant"],
        "\t",
        roles["both_uf"],
        "\t",
        roles["both_af"],
        "\t",
    )
    return (
        roles["user"],
        roles["assistant"],
        roles["both_uf"],
        roles["both_af"],
    )


def extract_example_prompt(s, e_word):
    split_text = s.lower().split()
    e_word_id = split_text.index(e_word)
    return " ".join(split_text[e_word_id - 5 : e_word_id + 6])


def extract_example_persona(s, e_word, wc=False):
    if not wc:
        s = get_persona_convo(s)
    for turn in s:
        split_text = turn["content"].lower().split()
        if e_word in split_text:
            e_word_id = split_text.index(e_word)
            return (
                turn["role"],
                " ".join(split_text[e_word_id - 5 : e_word_id + 6]),
            )
    # s = " ".join([t["content"] for t in ]).lower()
    # split_text = s.split()
    # # print(split_text, e_word)
    # e_word_id = split_text.index(e_word)
    # return " ".join(split_text[)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--long", action="store_true", help="Whether to have long replies"
    # )
    parser.add_argument(
        "--advice",
        action="store_true",
        help="Whether to focus on advice rather than conversation history",
    )
    parser.add_argument(
        "--wc",
        action="store_true",
        help="Whether to concatenate wildchat or regular results",
    )
    args = parser.parse_args()
    if args.advice:
        df_prompts = pd.read_pickle("data/base_data_prompt.gz")
        df_prompts = df_prompts.loc[
            df_prompts["prompt_type"] == "original_post"
        ]
    elif args.wc:
        open_least_sycophantic = Counter()
        open_most_sycophantic = Counter()
        closed_least_sycophantic = Counter()
        closed_most_sycophantic = Counter()
        with open("data/wildchat_career.pkl", "rb") as infile:
            career = pickle.load(infile)
        with open("data/wildchat_creative.pkl", "rb") as infile:
            creative = pickle.load(infile)
        with open("data/wildchat_random.pkl", "rb") as infile:
            random = pickle.load(infile)
        data = {
            "persona_type": ["career"] * len(career)
            + ["creative"] * len(creative)
            + ["random"] * len(random),
            "persona_id": list(range(len(career)))
            + list(range(len(creative)))
            + list(range(len(random))),
            "persona_text": career + creative + random
            # [
            #     " ".join([t["content"] for t in c]) for c in career + creative + random
            # ]
            ,
        }
        df_personas = pd.DataFrame(data)
    else:
        df_personas = pd.read_pickle("data/base_data_persona.gz")
    if os.path.isfile(
        f"data/marked_words_{'wc_' if args.wc else ''}{'adv_' if args.advice else ''}least.pkl"
    ):
        with open(
            f"data/marked_words_{'wc_' if args.wc else ''}{'adv_' if args.advice else ''}least.pkl",
            "rb",
        ) as infile:
            least_sycophantic = pickle.load(infile)
        with open(
            f"data/marked_words_{'wc_' if args.wc else ''}{'adv_' if args.advice else ''}most.pkl",
            "rb",
        ) as infile:
            most_sycophantic = pickle.load(infile)

        if not args.advice:
            print("Least sycophantic")
            print(least_sycophantic.most_common())
            for w in least_sycophantic.most_common():
                if (
                    w[0] != ""
                    and w[0] != "gordonfreemanftw"
                    and w[0] != "yesnon"
                    and w[0] != "systemsn"
                    # and w[0] != "dr"
                    # and w[0] != "rosehulman"
                    # and w[0] != "childrens"
                    # and w[0] != "faithbased"
                ):
                    w0 = (
                        w[0]
                        .replace("faithbased", "faith-based")
                        .replace("childrens", "children's")
                        .replace("rosehulman", "rose-hulman")
                        .replace("shellnix", "shell.nix")
                        .replace("management", "management/n")
                    )
                    if w0 == "dr":
                        w0 = "dr."
                    print(w[0])
                    print(
                        w[0],
                        [
                            extract_example_persona(
                                df_personas.loc[
                                    df_personas["persona_text"]
                                    .astype(str)
                                    .str.lower()
                                    .str.contains(" " + w0 + " ")
                                ]["persona_text"].iloc[0],
                                w0,
                                wc=args.wc,
                            )
                        ],
                        extract_roles(
                            df_personas.loc[
                                df_personas["persona_text"]
                                .astype(str)
                                .str.lower()
                                .str.contains(" " + w0 + " ")
                            ]["persona_text"].tolist(),
                            w0,
                            wc=args.wc,
                        ),
                    )
            print("Most sycophantic")
            print(most_sycophantic.most_common())
            for w in most_sycophantic.most_common():
                if w[0] != "":
                    try:
                        print(
                            w[0],
                            [
                                extract_example_persona(
                                    df_personas.loc[
                                        df_personas["persona_text"]
                                        .astype(str)
                                        .str.lower()
                                        .str.contains(" " + w[0] + " ")
                                    ]["persona_text"].iloc[0],
                                    w[0],
                                    wc=args.wc,
                                )
                            ],
                            extract_roles(
                                df_personas.loc[
                                    df_personas["persona_text"]
                                    .astype(str)
                                    .str.lower()
                                    .str.contains(" " + w[0] + " ")
                                ]["persona_text"].tolist(),
                                w[0],
                                wc=args.wc,
                            ),
                        )
                    except:
                        print(w[0])

            if args.wc:
                with open(
                    f"data/marked_words_open_wc_least.pkl", "rb"
                ) as infile:
                    least_sycophantic_open = pickle.load(infile)
                with open(
                    f"data/marked_words_open_wc_most.pkl", "rb"
                ) as infile:
                    most_sycophantic_open = pickle.load(infile)

                with open(
                    f"data/marked_words_closed_wc_least.pkl", "rb"
                ) as infile:
                    least_sycophantic_closed = pickle.load(infile)
                with open(
                    f"data/marked_words_closed_wc_most.pkl", "rb"
                ) as infile:
                    most_sycophantic_closed = pickle.load(infile)

                words = set(
                    [
                        w[0]
                        for w in least_sycophantic_open.most_common()
                        + most_sycophantic_open.most_common()
                        + least_sycophantic_closed.most_common()
                        + most_sycophantic_closed.most_common()
                    ]
                )
                for word in words:
                    if (
                        sum(
                            [
                                c != 0
                                for c in (
                                    least_sycophantic_open[word],
                                    most_sycophantic_open[word],
                                    least_sycophantic_closed[word],
                                    most_sycophantic_closed[word],
                                )
                            ]
                        )
                        > 1
                        and (
                            least_sycophantic_open[word]
                            + least_sycophantic_closed[word]
                        )
                        > 0
                        and (
                            most_sycophantic_open[word]
                            + most_sycophantic_closed[word]
                        )
                        > 0
                    ):
                        print(
                            word,
                            least_sycophantic_open[word],
                            most_sycophantic_open[word],
                            least_sycophantic_closed[word],
                            most_sycophantic_closed[word],
                        )

                print("Least sycophantic - Open models")
                print(least_sycophantic_open.most_common())
                print("Most sycophantic - Open models")
                print(most_sycophantic_open.most_common())
                print("Least sycophantic - Closed models")
                print(least_sycophantic_closed.most_common())
                print("Most sycophantic - Closed models")
                print(most_sycophantic_closed.most_common())

        else:
            print("Least sycophantic")
            print(least_sycophantic.most_common())
            for w in least_sycophantic.most_common():
                if w[0] != "":
                    print(
                        w[0],
                        [
                            extract_example_prompt(s, w[0])
                            for s in df_prompts.loc[
                                df_prompts["prompt_text"]
                                .str.lower()
                                .str.contains(" " + w[0] + " ")
                            ]["prompt_text"].iloc[:3]
                        ],
                    )
            print("Most sycophantic")
            print(most_sycophantic.most_common())
            for w in most_sycophantic.most_common():
                if w[0] != "":
                    print(
                        w[0],
                        [
                            extract_example_prompt(s, w[0])
                            for s in df_prompts.loc[
                                df_prompts["prompt_text"]
                                .str.lower()
                                .str.contains(" " + w[0] + " ")
                            ]["prompt_text"].iloc[:3]
                        ],
                    )
        # cloud_least = WordCloud(
        #     background_color="white"
        # ).generate_from_frequencies(least_sycophantic)
        # plt.imshow(cloud_least, interpolation="bilinear")
        # plt.axis("off")
        # plt.savefig("figures/least_sycophantic.png", dpi=600)
        # cloud_most = WordCloud(
        #     background_color="white"
        # ).generate_from_frequencies(most_sycophantic)
        # plt.imshow(cloud_most, interpolation="bilinear")
        # plt.axis("off")
        # plt.savefig("figures/most_sycophantic.png", dpi=600)
        # elif args.long and os.path.isfile(
        #     "data/marked_words_long_history_least.pkl"
        # ):
        #     with open("data/marked_words_long_advice_least.pkl", "rb") as infile:
        #         least_sycophantic = pickle.load(infile)
        #     with open("data/marked_words_long_advice_most.pkl", "rb") as infile:
        #         most_sycophantic = pickle.load(infile)

        #     print("Least sycophantic")
        #     print(least_sycophantic.most_common(20))
        #     print("Most sycophantic")
        #     print(most_sycophantic.most_common(20))

        #     with open("data/marked_words_long_history_least.pkl", "rb") as infile:
        #         least_negative = pickle.load(infile)
        #     with open("data/marked_words_long_history_most.pkl", "rb") as infile:
        #         most_negative = pickle.load(infile)

        #     print("Least negative")
        #     print(least_negative.most_common(20))
        #     print("Most negative")
        #     print(most_negative.most_common(20))
        # else:
        #     if args.long:
        #         least_sycophantic_advice = Counter()
        #         most_sycophantic_advice = Counter()
        #         for model in tqdm(all_models):
        #             if not os.path.isfile(f"results/{model}_long_results.pkl"):
        #                 continue
        #             df_long = pd.read_pickle(f"results/{model}_long_results.pkl")
        #             df_short = pd.read_pickle(f"results/{model}_results.pkl")
        #             least_s, most_s = get_marked_words_long_advice(
        #                 df_short, df_long, 200
        #             )
        #             del df_long, df_short
        #             least_s = [t[0] for t in least_s]
        #             most_s = [t[0] for t in most_s]
        #             least_sycophantic_advice.update(least_s)
        #             most_sycophantic_advice.update(most_s)
        #         with open(
        #             "data/marked_words_long_advice_least.pkl", "wb"
        #         ) as outfile:
        #             pickle.dump(least_sycophantic_advice, outfile)
        #         with open(
        #             "data/marked_words_long_advice_most.pkl", "wb"
        #         ) as outfile:
        #             pickle.dump(most_sycophantic_advice, outfile)

        #         least_negative_advice = Counter()
        #         most_negative_advice = Counter()
        #         df_personas = pd.read_pickle("data/base_data_persona.gz")
        #         for model in tqdm(all_models):
        #             if not os.path.isfile(
        #                 f"results/{model}_long_results_sentiment.pkl"
        #             ):
        #                 continue
        #             df_long_sent = pd.read_pickle(
        #                 f"results/{model}_long_results_sentiment.pkl"
        #             )
        #             least_n, most_n = get_marked_words_long_history(
        #                 df_long_sent, df_personas, 200
        #             )
        #             del df_long_sent
        #             least_n = [t[0] for t in least_n]
        #             most_n = [t[0] for t in most_n]
        #             least_negative_advice.update(least_n)
        #             most_negative_advice.update(most_n)

        #         del df_personas
        #         with open(
        #             "data/marked_words_long_history_least.pkl", "wb"
        #         ) as outfile:
        #             pickle.dump(least_negative_advice, outfile)
        #         with open(
        #             "data/marked_words_long_history_most.pkl", "wb"
        #         ) as outfile:
        #             pickle.dump(most_negative_advice, outfile)

    else:
        least_sycophantic = Counter()
        most_sycophantic = Counter()
        if args.wc:
            n = 30 if not args.advice else 100
            models = wc_models
            if not args.advice:
                df_personas["persona_text"] = df_personas[
                    "persona_text"
                ].astype(str)
        else:
            n = 200 if not args.advice else 100
            models = all_models

        for model in tqdm(models):
            df = pd.read_pickle(
                f"results/{model}_{'wc_' if args.wc else ''}results.pkl"
            )
            if args.advice:
                least_s, most_s = get_marked_words_short_advice(
                    df, df_prompts, n
                )
            else:
                least_s, most_s = get_marked_words_short(df, df_personas, n)
            least_s = [t[0] for t in least_s]
            most_s = [t[0] for t in most_s]
            least_sycophantic.update(least_s)
            most_sycophantic.update(most_s)
            if args.wc and not args.advice:
                if "gpt" in model or "gemini" in model:
                    closed_least_sycophantic.update(least_s)
                    closed_most_sycophantic.update(most_s)
                else:
                    open_least_sycophantic.update(least_s)
                    open_most_sycophantic.update(most_s)
            del df
        if args.advice:
            del df_prompts
        else:
            del df_personas

        with open(
            f"data/marked_words_{'wc_' if args.wc else ''}{'adv_' if args.advice else ''}least.pkl",
            "wb",
        ) as outfile:
            pickle.dump(least_sycophantic, outfile)
        with open(
            f"data/marked_words_{'wc_' if args.wc else ''}{'adv_' if args.advice else ''}most.pkl",
            "wb",
        ) as outfile:
            pickle.dump(most_sycophantic, outfile)

        if args.wc and not args.advice:
            with open(
                f"data/marked_words_closed_wc_{'adv_' if args.advice else ''}least.pkl",
                "wb",
            ) as outfile:
                pickle.dump(closed_least_sycophantic, outfile)
            with open(
                f"data/marked_words_closed_wc_{'adv_' if args.advice else ''}most.pkl",
                "wb",
            ) as outfile:
                pickle.dump(closed_most_sycophantic, outfile)

            with open(
                f"data/marked_words_open_wc_{'adv_' if args.advice else ''}least.pkl",
                "wb",
            ) as outfile:
                pickle.dump(open_least_sycophantic, outfile)
            with open(
                f"data/marked_words_open_wc_{'adv_' if args.advice else ''}most.pkl",
                "wb",
            ) as outfile:
                pickle.dump(open_most_sycophantic, outfile)
