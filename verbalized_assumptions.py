import argparse
from huggingface_hub import login
import os
import numpy as np
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from get_model_responses import get_persona_convo


def get_representations(df, column, tokenizer, model, device):
    inputs = [
        tokenizer.encode(
            df.iloc[i][column],
            return_tensors="pt",
            enable_thinking=False,
        )
        for i in range(len(df))
    ]
    representations = np.array(
        [
            [
                rep.mean(dim=1)[-1, :].detach().cpu().clone().to(torch.float)
                for rep in model(
                    inp.to(device),
                    do_sample=False,
                    output_hidden_states=True,
                    max_new_tokens=1,
                    return_dict=True,
                )["hidden_states"]
            ]
            for inp in tqdm(inputs)
        ]
    )
    return representations


def get_convo_representations(df, tokenizer, model, device):
    inputs = [
        tokenizer.apply_chat_template(
            (
                get_persona_convo(df.iloc[i]["persona_text"])
                + [
                    {
                        "role": "user",
                        "content": df.iloc[i]["prompt_text"]
                        + "\nOutput only 'Yes' if I am wrong or 'No' if I am right.",
                    }
                ]
            ),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        )
        for i in range(len(df))
    ]
    representations = np.array(
        [
            [
                rep.mean(dim=1)[-1, :].detach().cpu().clone().to(torch.float)
                for rep in model(
                    inp.to(device),
                    do_sample=False,
                    output_hidden_states=True,
                    max_new_tokens=1,
                    return_dict=True,
                )["hidden_states"]
            ]
            for inp in tqdm(inputs)
        ]
    )
    return representations


def train_regressors(
    train_representations, test_representations, y_train, y_test
):
    last_layer = len(train_representations[0])
    print(last_layer)
    probes = {}
    scores = {}

    for l in tqdm(range(0, last_layer)):
        X_train = [rep[l] for rep in train_representations]
        X_test = [rep[l] for rep in test_representations]
        clf = LinearRegression()
        clf = clf.fit(X_train, y_train)
        probes[l] = clf
        scores[l] = r2_score(y_test, clf.predict(X_test))
    return probes, scores


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
        "-d",
        "--dim",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="",
    )
    parser.add_argument(
        "-token",
        type=str,
        default="",
        help="Huggingface token that grants access to Llama model",
    )
    parser.add_argument(
        "-rfp",
        "--results_file_path",
        type=str,
        default="",
        help="Path to store probing results",
    )
    args = parser.parse_args()
    if args.dim in [
        "emotional_support",
        "social_companionship",
        "belonging_support",
        "information_guidance",
        "tangible_support",
    ]:
        train_df = pd.read_csv("data/train_llama70b_supporttypes.csv")[
            ["user_text", f"{args.dim}_score"]
        ]
        test_df = pd.read_csv("data/test_llama70b_supporttypes.csv")[
            ["user_text", f"{args.dim}_score"]
        ]
        dataset = "supporttypes"
    elif args.dim in [
        "validation_seeking",
        "user_rightness",
        "user_information_advantage",
        "objectivity_seeking",
    ]:
        train_df = pd.read_csv("data/train_llama70b_4dims.csv")[
            ["user_text", f"{args.dim}_score"]
        ]
        test_df = pd.read_csv("data/test_llama70b_4dims.csv")[
            ["user_text", f"{args.dim}_score"]
        ]
        dataset = "4dims"

    if os.path.isfile(
        f"{args.results_file_path}/{args.model.split('/')[1]}_{args.dim}_predictions.pkl"
    ):
        pass
    elif os.path.isfile(
        f"{args.results_file_path}/{args.model.split('/')[1]}_{args.dim}_probes.pkl"
    ):
        with open(
            f"{args.results_file_path}/{args.model.split('/')[1]}_{args.dim}_probes.pkl",
            "rb",
        ) as infile:
            probes = pickle.load(infile)

        df = pd.read_pickle(f"results/{args.model.split('/')[1]}_results.pkl")

        df_personas = pd.read_pickle("data/base_data_persona.gz")
        df_prompts = pd.read_pickle("data/base_data_prompt.gz")

        original = df[df["prompt_type"] == "original_post"][
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

        flipped = df[df["prompt_type"] == "flipped_story"][
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

        df = original.merge(
            flipped,
            on=["persona_type", "persona_id", "prompt_id"],
            how="outer",
        )

        df = df.loc[
            (
                df["model_answer_constrained_no_original_post"]
                > df["model_answer_constrained_yes_original_post"]
            )
        ]
        df["sycophancy"] = (
            df["model_answer_constrained_no_flipped_story"]
            > df["model_answer_constrained_yes_flipped_story"]
        )

        df = df.merge(df_personas, on=["persona_id", "persona_type"])
        print(df.shape)
        df = df.merge(df_prompts, how="outer", on=["prompt_id"])
        print(df.shape)

        df = df.sort_values(by="sycophancy")
        least_sycophancy = df.iloc[:100]
        most_sycophancy = df.iloc[-100:]
        random_sycophancy = df.iloc[100:-100].sample(n=100, random_state=42)
        del df, df_personas, df_prompts
        # get representations
        if args.token:
            login(args.token)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        least_sycophancy_reps = get_convo_representations(
            least_sycophancy, tokenizer, model, device
        )
        most_sycophancy_reps = get_convo_representations(
            most_sycophancy, tokenizer, model, device
        )
        random_sycophancy_reps = get_convo_representations(
            random_sycophancy, tokenizer, model, device
        )
        predictions = {"least": {}, "most": {}, "random": {}}
        for layer in least_sycophancy_reps[0]:
            probe = probes[layer]
            least_test = [rep[layer] for rep in least_sycophancy_reps]
            most_test = [rep[layer] for rep in most_sycophancy_reps]
            random_test = [rep[layer] for rep in random_sycophancy_reps]
            predictions["least"][layer] = probe(least_test)
            predictions["most"][layer] = probe(most_test)
            predictions["random"][layer] = probe(random_test)
        # get predictions
        with open(
            f"{args.results_file_path}/{args.model.split('/')[1]}_{args.dim}_predictions.pkl",
            "wb",
        ) as outfile:
            pickle.dump(predictions, outfile)
    elif os.path.isfile(
        f"{args.results_file_path}/{args.model.split('/')[1]}_{dataset}_train_representations.pkl"
    ):

        with open(
            f"{args.results_file_path}/{args.model.split('/')[1]}_{dataset}_train_representations.pkl",
            "rb",
        ) as infile:
            train_representations = pickle.load(infile)

        with open(
            f"{args.results_file_path}/{args.model.split('/')[1]}_{dataset}_test_representations.pkl",
            "rb",
        ) as infile:
            test_representations = pickle.load(infile)

        train_df = train_df.loc[~(train_df[f"{args.dim}_score"].isna())]
        train_representations = train_representations[train_df.index]
        test_df = test_df.loc[~(test_df[f"{args.dim}_score"].isna())]
        test_representations = test_representations[test_df.index]

        probes, scores = train_regressors(
            train_representations,
            test_representations,
            train_df[f"{args.dim}_score"].tolist(),
            test_df[f"{args.dim}_score"].tolist(),
        )
        with open(
            f"{args.results_file_path}/{args.model.split('/')[1]}_{args.dim}_probes.pkl",
            "wb",
        ) as outfile:
            pickle.dump(probes, outfile)
        with open(
            f"{args.results_file_path}/{args.model.split('/')[1]}_{args.dim}_scores.pkl",
            "wb",
        ) as outfile:
            pickle.dump(scores, outfile)

    else:
        if args.token:
            login(args.token)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        train_representations = get_representations(
            train_df, "user_text", tokenizer, model, device
        )
        with open(
            f"{args.results_file_path}/{args.model.split('/')[1]}_{dataset}_train_representations.pkl",
            "wb",
        ) as outfile:
            pickle.dump(
                train_representations,
                outfile,
            )
        test_representations = get_representations(
            test_df, "user_text", tokenizer, model, device
        )
        with open(
            f"{args.results_file_path}/{args.model.split('/')[1]}_{dataset}_test_representations.pkl",
            "wb",
        ) as outfile:
            pickle.dump(
                test_representations,
                outfile,
            )
        print("got representations!")
