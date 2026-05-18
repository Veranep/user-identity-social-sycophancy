import argparse
import os
from tqdm import tqdm
import pandas as pd

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
    df = pd.read_pickle(
        f"results/{args.model.split('/')[-1]}{'_long' if args.long else ''}_results.pkl"
    )
    if "persona_text_cat" not in df:
        values = []
        for convo in tqdm(df["persona_text"].values):
            string_convo = (
                convo.strip()
                .replace("\n", " ")
                .replace('12"x12"', "12x12 inches")
                .replace("\u00D7", "x")
                .replace('1/2"', "1/2 inch")
                .replace('1"x2"', "1x2 inches")
                .replace('3/4"', "3/4 inch")
                .replace('12" x 8"', "12 x 8 inches")
                .replace('"assistant":', '"assistant",')
                .split("  // Note:")[0]
            )
            if string_convo[-4:] == "]} ]":
                string_convo = string_convo[:-4] + "}]"
            elif string_convo[-4:] == "   ]":
                string_convo = string_convo[:-4] + "}]"
            if string_convo[-3:] != '"}]':
                if string_convo[-1] == ".":
                    string_convo += '"}]'
                elif string_convo[-3:] == '"]}':
                    string_convo = string_convo[:-3] + '"}]'
                elif string_convo[-3:] == "} ]":
                    string_convo = string_convo[:-3] + "}]"
                elif string_convo[-3:] == "]}]":
                    string_convo = string_convo[:-3] + "}]"
                elif string_convo[-3:] == "}]}":
                    string_convo = string_convo[:-3] + "}]"
                elif string_convo[-3:] == "}}]":
                    string_convo = string_convo[:-3] + "}]"
                if string_convo[-3:] == ".}]":
                    string_convo = string_convo[:-3] + '."}]'
            persona_convo = eval(string_convo)
            for t in range(len(persona_convo)):
                if (
                    len(persona_convo) > 2
                    and persona_convo[t]["role"] == "assistant"
                    and persona_convo[t - 1]["role"] == "assistant"
                ):
                    persona_convo[t - 1]["content"] = (
                        persona_convo[t - 1]["content"]
                        + " "
                        + persona_convo[t]["content"]
                    )
                    del persona_convo[t]
            values.append(" ".join([t["content"] for t in persona_convo]))
        df["persona_text_cat"] = values

        df.to_pickle(
            f"results/{args.model.split('/')[-1]}{'_long' if args.long else ''}_results.pkl"
        )
