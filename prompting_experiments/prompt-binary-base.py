"""
Run binary abuse classification (ABU / NOTABU) using one and few-shot prompting.

Uses a local LLaMA model (via OpenAI API interface) to classify tweets based on prompt.
Supports shot setups (P1–P4) and prompt modes ('base' or 'def'), and temperatures ('temp').
Saves model predictions and optionally evaluates them.
"""

import pandas as pd
import json
import re
import os
import ast
import argparse
from openai import OpenAI
from sklearn.metrics import classification_report

# Connect to local LLaMA server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

def query_LLM(model_client, prompt, temp=0):
    """
    Send a prompt to the local model and return the generated response.
    """
    history = [{"role": "user", "content": prompt}]
    completion = model_client.chat.completions.create(
        model="local-model",
        messages=history,
        temperature=temp,
        stream=False,
    )
    return completion.choices[0].message.content

# Pool of few-shot examples (label, text)
EXAMPLES_ALL = [
    # ABU
    ("ABU", "@USER you are a lying corrupt traitor!!! Nobody wants to hear anymore of your lies!!! #DeepStateCorruption URL"),  #EXP TIN/IND
    ("ABU", "@USER Gun Control means using both hands. And in this case means acquiring the correct target. You can’t fix stupid."),  #IMP UNT
    ("ABU", "@USER “the sex symbol of rap” LMAOOO BITCH YOUre  FAT aS FUCK ANYWAYS you tryna fight?"),  #EXP TIN/GRP
    ("ABU", "10kids from different women from this American jobless men? Nigerian men you are the bomb!!!"),  #IMP TIN/OTH
    ("ABU", "MAGA are PERVERTS AND DEMONS-PERIOD! NOTHING IS SACRED OR INNOCENT WITH THESE MUTTS OF SATAN URL"),  #EXP TIN/OTH
    ("ABU", "4 out of 10 British people are basically full-on racists. 4 out of 10 voters vote for the Conservatives. Coincidence!???!???"),  #IMP TIN/GRP
    ("ABU", "@USER What the fuck game are you watching?"),  #EXP UNT
    ("ABU", "@USER       Thank you for your fortitude and perseverance. Please give McConnell a kick in the butt from some of us conservatives."),  #IMP TIN/IND
    ("ABU", "@USER If you go by anything other than he or she you are fucked in the head."),  #EXP TIN/IND
    ("ABU", "2 hours at midnight and she said she is not a call girl?  -Awkward!"),  #IMP UNT
    ("ABU", "@USER @USER The FASCISTS are Anti-American Communist Scum like yourself!! Now go Fuck Yourself you piece of human garbage!! USA #MAGA  ANTIFA are Terrorists"),  #EXP TIN/GRP
    ("ABU", "@USER Do you think she is afraid of pigs that are not the same colour as her? 😊"),  #IMP TIN/OTH

    # NOTABU
    ("NOTABU", "@USER What if knowing that abortion is at stake in November mobilizes pro life forces? Kavanaugh being denied makes it critical for conservatives to maintain the Senate. Because most (R) seats are not up for election it could be a blessing."),
    ("NOTABU", "(You are! My prayer is that you are blessed with someone you know like that also. Someone who feels your basic human rights are worth protecting. Especially your God given right to personal privacy.)"),
    ("NOTABU", "@USER You should reach out to Paul.berger@USER He is teaching & researching these classes. & he is fierce & great! V. Ethical."),
    ("NOTABU", "@USER You guys have featured splendidrainco's etsy page and she is being censored by etsy now. You can't find her site through the search engine you have to have a direct link. And according to her instagram post they have been basically leaving her out to dry."),
    ("NOTABU", "@USER Aww Bradley you are so cute"),
    ("NOTABU", "@USER So happy for you and your dad!!! My dad had the same kind of cancer but stage 4. He was diagnosed in 2016 and passed away this passed January. My mom is kicking leukemia’s ass since Nov. of 2015 and she is doing awesome!"), 
    ("NOTABU", "@USER @USER It's OK that you are confused. Paul Guenther is a proven DC and play caller. We are rebuilding and trying to remain competitive at the same time. We will have steady improvement"),
    ("NOTABU", "@USER I loved the House Bunny. And the remembering names voice was funny. My family and I watch Mom it’s great. I’m a recovered addict or moderately recovered if you are one who counts pot lol. It’s a very good show."),
    ("NOTABU", "@USER @USER But Google can give Beijing every Americans phone number. You’re kidding me right? MAGA 2020"),
    ("NOTABU", "@USER sorry you are feeling sick hope you get better soon."),
    ("NOTABU", "@USER That's not Southern California. It is considered Central California. Great how our gun control laws work here isn't it?"),
    ("NOTABU", "@USER It is so funny that she is scared of pigs. This could be a plot for your next children's book.")
]


def build_prompt(setup, batch, mode="base"):
    """
    Build a prompt for classification, optionally including examples and definitions.
    """
    n = len(batch)
    assert n > 0, "Batch cannot be empty."
    numbered_tweets = "\n".join([f"{j+1}. {t}" for j, t in enumerate(batch)])

    if mode == "def":
        base = """Classify the following texts into one of the following categories:
- abusive (ABU): Language that is explicitly or implicitly offensive, hurtful, or profane. This includes speech that debases, insults, or expresses intense negative emotion. This also includes subtle forms like sarcasm and passive-aggression.
- not abusive (NOTABU): Language that is polite, respectful or neutral, and free from explicit or implicit harmful intent.
"""
    else:
        base = """Classify the following texts into one of the following categories:
- abusive (ABU)
- not abusive (NOTABU)
"""

    shots = {
        "P1": 0,
        "P2": 1,
        "P3": 8,
        "P4": 12,
    }.get(setup, 0)

    if shots > 0:
        abu_examples = [t for t in EXAMPLES_ALL if t[0] == "ABU"][:shots] # add ABU examples
        notabu_examples = [t for t in EXAMPLES_ALL if t[0] == "NOTABU"][:shots] # add NOTABU examples

        if len(abu_examples) < shots or len(notabu_examples) < shots:
            raise ValueError("Not enough ABU or NOTABU examples available.")

        example_lines = []
        for label, text in abu_examples + notabu_examples:
            example_lines.append(f"{label}: {text}")

        base += "\nHere are a few examples of labeled texts:\n" + "\n".join(example_lines)

    base += f"""
Now classify the following texts. Always choose ONE label per text.
Output ONLY a valid Python list of {n} labels. Do NOT explain. Do NOT add anything else.

Texts:
{numbered_tweets}
"""

    return base

def main():
    """
    Run LLM-based classification with few-shot prompting and save predictions.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", choices=["P1", "P2", "P3", "P4"], required=True, help="Setup (P1–P4)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for the model (default: 0.0)")
    parser.add_argument("--mode", choices=["base", "def"], default="base", help="Prompt mode: 'base' or 'def' (default: base)")
    args = parser.parse_args()
    setup = args.setup
    temperature = args.temperature
    mode = args.mode

    # Load test data
    df = pd.read_csv("../data/test/OLID_AbuseEval_test.tsv", sep="\t")
    tweets = df["tweet"].tolist()
    print(f"▶ Running binary setup {setup} (mode={mode}) on {len(tweets)} tweets...")

    predictions = []
    batch_size = 5  # number of tweets per prompt
    valid_labels = {"ABU", "NOTABU"}

    # Process tweets in batches
    for i in range(0, len(tweets), batch_size):
        batch = tweets[i:i+batch_size]
        prompt = build_prompt(setup, batch, mode=mode)

        try:
            result = query_LLM(client, prompt, temp=temperature)
            print(f"\n=== MODEL OUTPUT ({i}) ===\n{result}\n")

            # Extract list of labels from model output
            match = re.search(r"\[[^\]]+\]", result)
            if match:
                raw_list = match.group(0)
                try:
                    labels = json.loads(raw_list)
                except json.JSONDecodeError:
                    # Fallback if JSON is not valid
                    fixed = re.sub(r'(?<=\[|\s|,)(ABU|NOTABU)(?=,|\s|\])', r'"\1"', raw_list)
                    labels = ast.literal_eval(fixed)

                labels = [label.strip().upper() for label in labels]

                # Validate output
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

    # Final check for consistency
    if len(predictions) != len(tweets):
        raise RuntimeError(f"Final mismatch: {len(predictions)} predictions vs. {len(tweets)} tweets")

    # Save predictions
    df = df.iloc[:len(predictions)]
    df["prediction"] = predictions
    os.makedirs("../predictions", exist_ok=True)
    pred_file = f"../predictions/binary-{args.mode}-{setup}-t{temperature:.2f}_predictions.csv"
    df.to_csv(pred_file, index=False)
    print(f"✅ Predictions saved to {pred_file}")

    # If gold labels are available, compute classification report
    if "abuse" in df.columns:
        label_map = {
            "EXP": "ABU",
            "IMP": "ABU",
            "NOTABU": "NOTABU"
        }
        if not df["abuse"].isin(label_map.keys()).all():
            print("❗ Found unexpected labels in 'abuse' column!")
        else:
            df["gold_binary"] = df["abuse"].map(label_map)
            report = classification_report(df["gold_binary"], df["prediction"], labels=["ABU", "NOTABU"])
            report_file = f"../predictions/binary-{args.mode}-{setup}-t{temperature:.2f}_report.txt"
            with open(report_file, "w") as f:
                f.write(f"Setup: {setup}, Mode: {mode}\n\n")
                f.write(report)
            print(f"📊 Report saved to {report_file}")
    else:
        print("⚠️ Column 'abuse' not found for evaluation.")

if __name__ == "__main__":
    main()
