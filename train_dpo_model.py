from datasets import Dataset
import pandas as pd

from trl import DPOTrainer


def preprocess_function(example):
    return {
        "prompt": [{"role": "user", "content": example["prompt"]}],
        "chosen": [{"role": "assistant", "content": example["chosen"]}],
        "rejected": [{"role": "assistant", "content": example["rejected"]}],
    }


if __name__ == "__main__":
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
                "response_0": "chosen",
                "response_1": "rejected",
            },
            errors="ignore",
        )
        train_df = pd.concat([train_df, df], ignore_index=True)
    ds = Dataset.from_pandas(train_df)
    ds = ds.map(preprocess_function)
    print(ds)

    trainer = DPOTrainer(
        model="meta-llama/Llama-3.1-8B-Instruct",
        train_dataset=ds,
    )
    trainer.train()
    trainer.save_model("results/dpo_model")
