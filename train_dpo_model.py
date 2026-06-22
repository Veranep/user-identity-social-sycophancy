import argparse
from datasets import Dataset
from huggingface_hub import login
import numpy as np
import pandas as pd

import torch

torch.backends.cuda.enable_cudnn_sdp(False)

from trl import DPOTrainer, DPOConfig


def preprocess_function(example):
    return {
        "prompt": [{"role": "user", "content": example["prompt"]}],
        "chosen": [{"role": "assistant", "content": example["chosen"]}],
        "rejected": [{"role": "assistant", "content": example["rejected"]}],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    train_df = pd.DataFrame([])
    for file_name, dimension in [
        ("train_ev_aita.csv", "ev"),
        ("train_indirect_aita.csv", "suggestive"),
        ("train_framing_aita.csv", "challenge"),
        ("train_framing_oeq.csv", "challenge"),
        ("train_ev_oeq.csv", "ev"),
        ("train_indirect_oeq.csv", "suggestive"),
        ("train_framing_ss.csv", "framing"),
    ]:
        needed_cols = [
            "prompt",
            "response_0",
            "response_1",
        ]
        df = pd.read_csv("dpo/" + file_name, index_col=0)[needed_cols]
        df = df.rename(
            columns={
                "response_0": "rejected",
                "response_1": "chosen",
            },
            errors="ignore",
        )
        train_df = pd.concat([train_df, df], ignore_index=True)
    ds = Dataset.from_pandas(train_df)
    ds = ds.map(preprocess_function)
    print(ds)

    del df
    del train_df

    config = DPOConfig(save_strategy="no")

    trainer = DPOTrainer(
        model="meta-llama/Llama-3.1-8B-Instruct", train_dataset=ds, args=config
    )
    trainer.train()
    trainer.save_model("results/dpo_model")
