"""
Run ternary abuse classification (EXP / IMP / NOTABU) using one and few-shot prompting.

Uses a local LLaMA model (via OpenAI API interface) to classify tweets based on prompt.
Supports multiple shots setups (P1–P4), prompt modes ('base' or 'def'), and temperatures ('temp').
Saves model predictions and optionally evaluates them.
"""

import pandas as pd
import json
import re
import os
import argparse
import ast
from openai import OpenAI
from sklearn.metrics import classification_report

# Connect to local LLaMA server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

def query_LLM(model_client, prompt, temp=0):
    """
    Query the LLM with a prompt and return the model response.

    Args:
        model_client: OpenAI-compatible client (e.g., local LLaMA).
        prompt (str): Prompt to send to the model.
        temp (float): Sampling temperature.

    Returns:
        str: Model's response (text only).
    """
    history = [{"role": "user", "content": prompt}]
    completion = model_client.chat.completions.create(
        model="local-model",
        messages=history,
        temperature=temp,
        stream=False,
    )
    return completion.choices[0].message.content

# Your labeled examples
EXAMPLES_ALL = [
    # EXP
    ("EXP", "@USER you are a lying corrupt traitor!!! Nobody wants to hear anymore of your lies!!! #DeepStateCorruption URL"), #TIN/IND
    ("EXP", "@USER “the sex symbol of rap” LMAOOO BITCH YOUre  FAT aS FUCK ANYWAYS you tryna fight?"), #TIN/GRP
    ("EXP", "MAGA are PERVERTS AND DEMONS-PERIOD! NOTHING IS SACRED OR INNOCENT WITH THESE MUTTS OF SATAN URL"), #TIN/OTH
    ("EXP", "@USER What the fuck game are you watching?"), #UNT
    ("EXP", "@USER If you go by anything other than he or she you are fucked in the head."), #TIN/IND
    ("EXP", "@USER @USER The FASCISTS are Anti-American Communist Scum like yourself!! Now go Fuck Yourself you piece of human garbage!! USA #MAGA  ANTIFA are Terrorists"), #TIN/GRP
    ("EXP", "@USER @USER Are you bipolar? You keep saying different shit to people commenting on what u said"), #TIN/OTH
    ("EXP", "@USER So what ur saying is that it is in fact chicken shit and you are eating chicken shit URL"), #UNT

    # IMP
    ("IMP", "@USER Gun Control means using both hands. And in this case means acquiring the correct target. You can’t fix stupid."), #UNT
    ("IMP", "10kids from different women from this American jobless men? Nigerian men you are the bomb!!!"),  #TIN/OTH
    ("IMP", "4 out of 10 British people are basically full-on racists. 4 out of 10 voters vote for the Conservatives. Coincidence!???!???"),  #TIN/GRP
    ("IMP", "@USER       Thank you for your fortitude and perseverance. Please give McConnell a kick in the butt from some of us conservatives."),  #TIN/IND
    ("IMP", "2 hours at midnight and she said she is not a call girl?  -Awkward!"), #UNT
    ("IMP", "@USER Do you think she is afraid of pigs that are not the same colour as her? 😊"), #TIN/OTH
    ("IMP", "@USER Love how fat they all are while chowing down on cake and cookies. That's the future liberals want."), #TIN/GRP
    ("IMP", "@USER Absolutely should be a nail in the coffin.... for Hillary"), #TIN/IND

    # NOTABU
    ("NOTABU", "@USER What if knowing that abortion is at stake in November mobilizes pro life forces? Kavanaugh being denied makes it critical for conservatives to maintain the Senate. Because most (R) seats are not up for election it could be a blessing."),
    ("NOTABU", "(You are! My prayer is that you are blessed with someone you know like that also. Someone who feels your basic human rights are worth protecting. Especially your God given right to personal privacy.)"),
    ("NOTABU", "@USER You should reach out to Paul.berger@USER He is teaching & researching these classes. & he is fierce & great! V. Ethical."),
    ("NOTABU", "@USER You guys have featured splendidrainco's etsy page and she is being censored by etsy now. You can't find her site through the search engine you have to have a direct link. And according to her instagram post they have been basically leaving her out to dry."),
    ("NOTABU", "@USER Aww Bradley you are so cute"),
    ("NOTABU", "@USER So happy for you and your dad!!! My dad had the same kind of cancer but stage 4. He was diagnosed in 2016 and passed away this passed January. My mom is kicking leukemia’s ass since Nov. of 2015 and she is doing awesome!"),
    ("NOTABU", "@USER @USER It's OK that you are confused. Paul Guenther is a proven DC and play caller. We are rebuilding and trying to remain competitive at the same time. We will have steady improvement"),
    ("NOTABU", "@USER I loved the House Bunny. And the remembering names voice was funny. My family and I watch Mom it’s great. I’m a recovered addict or moderately recovered if you are one who counts pot lol. It’s a very good show.")
]

def build_prompt(setup, batch, mode="base"):
    """
    Query the LLM with a prompt and return the model response.

    Args:
        model_client: OpenAI-compatible client (e.g., local LLaMA).
        prompt (str): Prompt to send to the model.
        temp (float): Sampling temperature.

    Returns:
        str: Model's response (text only).
    """
    n = len(batch)
    assert n > 0, "Batch cannot be empty."

    numbered_tweets = "\n".join([f"{j+1}. {t}" for j, t in enumerate(batch)])

    if mode == "def":
        base = """Classify the following texts into one of the following categories:
- explicit abuse (EXP): Language with direct and literal forms of abusive speech, such as slurs, profanity, and other clearly hostile or offensive expressions.
- implicit abuse (IMP): Language with subtle and indirect forms of abusive speech, such as passive-aggressiveness, sarcasm with harmful undertone, and other forms with implied harm.
- not abusive (NOTABU): Language that is polite, respectful or neutral, and free from explicit or implicit harmful intent.
"""
    else:
        base = """Classify the following texts into one of the following categories:
- explicit abuse (EXP)
- implicit abuse (IMP)
- not abusive (NOTABU)
"""

    shots = {
        "P1": 0,
        "P2": 1,
        "P3": 6,
        "P4": 8,
    }.get(setup, 0)

    if shots > 0:
        exp_examples = [t for t in EXAMPLES_ALL if t[0] == "EXP"][:shots]
        imp_examples = [t for t in EXAMPLES_ALL if t[0] == "IMP"][:shots]
        notabu_examples = [t for t in EXAMPLES_ALL if t[0] == "NOTABU"][:shots]

        few_shots = exp_examples + imp_examples + notabu_examples
        example_lines = [f"{label}: {text}" for label, text in few_shots]

        base += "\nHere are a few examples of labeled texts:\n" + "\n".join(example_lines)
        base += "\n\n"
    else:
        base += "\n"

    base += f"""Now classify the following texts. Always choose ONE label per text.
Output ONLY a valid Python list of {n} labels. Do NOT explain. Do NOT add anything else.

Texts:
{numbered_tweets}
"""

    return base

def parse_model_output(raw_list):
    """
    Parse the list of labels returned by the model.

    Args:
        raw_list (str): Raw model output containing a Python-style list.

    Returns:
        list: Parsed list of labels (e.g. ["EXP", "IMP", ...]).
    """
    try:
        return json.loads(raw_list)
    except json.JSONDecodeError:
        fixed = re.sub(r'(?<=\[|\s|,)(EXP|IMP|NOTABU)(?=,|\s|\])', r'"\1"', raw_list)
        try:
            return ast.literal_eval(fixed)
        except Exception as e:
            raise ValueError(f"Failed to parse model output: {raw_list}") from e

def main():
    """
    Run ternary abuse classification (EXP / IMP / NOTABU) using few-shot prompting.
    Saves model predictions and evaluation report (if labels exist).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", choices=["P1", "P2", "P3", "P4"], required=True, help="Setup (P1–P4)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for the model (default: 0.0)")
    parser.add_argument("--mode", choices=["base", "def"], default="base", help="Prompt mode (default: base)")
    args = parser.parse_args()
    setup = args.setup
    temperature = args.temperature
    mode = args.mode

    # Load test file
    df = pd.read_csv("../data/test/OLID_AbuseEval_test.tsv", sep="\t")
    tweets = df["tweet"].tolist()
    print(f"▶ Running ternary setup {setup} (mode={mode}) on {len(tweets)} tweets...")

    predictions = []
    batch_size = 5  # LLM batch size

    valid_labels = {"EXP", "IMP", "NOTABU"}

    for i in range(0, len(tweets), batch_size):
        batch = tweets[i:i+batch_size]
        prompt = build_prompt(setup, batch, mode=mode)

        try:
            # Query LLM and extract prediction labels
            result = query_LLM(client, prompt, temp=temperature)
            print(f"\n=== MODEL OUTPUT ({i}) ===\n{result}\n")
            match = re.search(r"\[[^\]]+\]", result)
            if match:
                raw_list = match.group(0)
                labels = parse_model_output(raw_list)
                labels = [label.strip().upper() for label in labels]

                if len(labels) != len(batch):
                    raise ValueError(f"Mismatch: expected {len(batch)} labels, got {len(labels)}")

                if not all(label in valid_labels for label in labels):
                    raise ValueError(f"Invalid label(s): {labels}")

                predictions.extend(labels)
            else:
                raise ValueError("No list found in model output")
        except Exception as e:
            print(f"⚠️ Error on batch {i}: {e}")
            predictions.extend(["ERROR"] * len(batch))

    # Final check and attach predictions
    if len(predictions) != len(tweets):
        raise RuntimeError(f"Final mismatch: {len(predictions)} predictions vs. {len(tweets)} tweets")

    df = df.iloc[:len(predictions)]
    df["prediction"] = predictions

    # Save predictions
    os.makedirs("../predictions", exist_ok=True)
    pred_file = f"../predictions/ternary-{args.mode}-{setup}-t{temperature:.2f}_predictions.csv"
    report_file = f"../predictions/ternary-{args.mode}-{setup}-t{temperature:.2f}_report.txt"
    df.to_csv(pred_file, index=False)
    print(f"✅ Predictions saved to {pred_file}")

    # Generate evaluation report
    if "abuse" in df.columns:
        report = classification_report(df["abuse"], df["prediction"], labels=["EXP", "IMP", "NOTABU"])
        with open(report_file, "w") as f:
            f.write(f"Setup: {setup}, Mode: {mode}\n\n")
            f.write(report)
        print(f"📊 Report saved to {report_file}")
    else:
        print("⚠️ Column 'abuse' not found for evaluation.")

if __name__ == "__main__":
    main()
