"""
Run binary abuse classification (ABU / NOTABU) using one and few-shot prompting.

Uses a local LLaMA model (via OpenAI API interface) to classify tweets based on prompt-guided reasoning.
Supports multiple shot setups (P1–P4) and prompt modes ('standard', 'targeted', or 'IHC_labels'), and temperatures ('temp').
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

### Connect to local LLaMA server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

def query_LLM(model_client, prompt, temp=0):
    """
    Send a prompt to the local LLaMA server and return the response.
    """
    history = [{"role": "user", "content": prompt}]
    completion = model_client.chat.completions.create(
        model="local-model",
        messages=history,
        temperature=temp,
        stream=False,
        # top_p=0.0
    )
    return completion.choices[0].message.content

### Pool of few-shot examples (label, text)
EXAMPLES_ALL = [
    # EXP 12
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

    # NOTABU 12
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
    ("NOTABU", "@USER It is so funny that she is scared of pigs. This could be a plot for your next children's book."),
]

### REASONING 
REASONING_BASE = [
    # ABU 12
    "This message attacks a person directly using insults and strong anger.",
    "This message suggests violence and ends with an insult.",
    "This message includes profanity, body shaming, and a threat. It is rude and aggressive.",
    "This message mocks Nigerian men using stereotypes and sarcasm. It is insulting and racist.",
    "This message uses extreme insults to attack a political movement. It is hateful and hostile.",
    "This message links a group to racism with a harmful stereotype and ends with a suggestive question. It is offensive.",
    "This message asks a question in a disrespectful way and includes profanity.",
    "This message suggests physical aggression against someone. Even if partly joking, it is still hostile.",
    "This message attacks people for their gender identity using profanity. It's demeaning.",
    "This message implies a degrading assumption through a mocking question. It's suggestive and disrespectful.",
    "This message is extremely aggressive and dehumanizing. It includes personal and political attacks.",
    "This message implies a racial insult in a sarcastic tone and compares someone to animals.",

    # NOTABU 12
    "This message shares a political opinion without insults or hate.",
    "This message includes wishes and prayers. It is kind and respectful toward someone.",
    "This message gives a helpful suggestion and ends by praising the suggested teacher.",
    "This message expresses light frustration about how someone is treated but remains respectful.",
    "This message is a simple, kind compliment.",
    "This message shows support and tells a personal story about the health of family members.",
    "This message shares a sports opinion in an affirming and supportive way.",
    "This message talks about TV shows in a light and personal way.",
    "This message expresses political skepticism while staying polite.",
    "This message shows support and wishes someone good health.",
    "This message is a sarcastic political opinion. It expresses light criticism but remains respectful.",
    "This message discusses something funny and playfully suggests writing a book about it."
]

### REASONING TARGET
REASONING_TARGET = [
    # ABU 12
    "This message targets an individual directly using insults and strong anger.",
    "This message suggests violence and ends with an insult, but it is not directly targeted.",
    "This message includes body shaming that is targeted toward a group.",
    "This message targets Nigerian men using stereotypes and sarcasm.",
    "This message uses extreme insults to target a political movement.",
    "This message targets a group by linking it to racism through a harmful stereotype and a suggestive question.",
    "This message asks a question in a disrespectful way and includes profanity, but remains untargeted.",
    "This message suggests physical aggression targeted at an individual.",
    "This message targets an individual by attacking their gender identity using profanity.",
    "This message implies a degrading assumption through a mocking question, but remains untargeted.",
    "This message uses personal and political insults to target a group.",
    "This message implies a racial insult in a sarcastic way, targeting people based on skin color.",

    # NOTABU 12
    "This message shares a political opinion without any target.",
    "This message includes wishes and prayers. It is kind and respectful toward someone, but not targeted.",
    "This message gives a helpful suggestion and ends by praising the suggested teacher, but remains untargeted.",
    "This message expresses light frustration about how someone is treated but remains untargeted.",
    "This message is a simple, kind compliment and stays untargeted.",
    "This message shows support and tells a personal story about the health of family members. It does not include a target.",
    "This message shares a sports opinion in an affirming and supportive way without a target.",
    "This message talks about TV shows in a light and personal way without a target.",
    "This message expresses political skepticism while staying polite and untargeted.",
    "This message shows support and wishes someone good health, and stays untargeted.",
    "This message is a sarcastic political opinion. It expresses light criticism but remains untargeted.",
    "This message discusses something funny and playfully suggests writing a book about it, with no target.",
]

### REASONING IHC
REASONING_IHC = [
    # ABU 12
    "This message attacks a person directly using insults and strong anger. It includes threats and intimidation.",
    "This message suggests violence and ends with an insult. It includes incitement to violence and irony.",
    "This message includes profanity, body shaming, and a threat. It includes inferiority language and intimidation.",
    "This message mocks Nigerian men using stereotypes and sarcasm. It includes stereotypes and misinformation.",
    "This message uses extreme insults to attack a political movement. It includes threats and dehumanizing language.",
    "This message links a group to racism with a harmful stereotype and ends with a suggestive question. It includes stereotypes and misinformation.",
    "This message asks a question in a disrespectful way and includes profanity. It includes irony and mild intimidation.",
    "This message suggests physical aggression against someone. Even if partly joking, it includes incitement to violence.",
    "This message attacks people for their gender identity using profanity. It includes inferiority language and intimidation.",
    "This message implies a degrading assumption through a mocking question. It includes irony and inferiority language.",
    "This message is extremely aggressive and dehumanizing. It includes threats, intimidation, and hate toward a political movement.",
    "This message implies a racial insult in a sarcastic tone and compares someone to animals. It includes stereotypes and inferiority language.",

    # NOTABU 12
    "This message shares a political opinion without insults or hate. It does not include any harmful framing.",
    "This message includes wishes and prayers. It is kind and respectful toward someone. It does not include threats, stereotypes, or other harmful speech.",
    "This message gives a helpful suggestion and ends by praising the suggested teacher. It does not include any harmful content.",
    "This message expresses light frustration about how someone is treated but remains respectful. It does not include stereotypes or insults.",
    "This message is a simple, kind compliment. It contains no negativity or bias.",
    "This message shows support and tells a personal story about the health of family members. It is empathetic and non-hostile.",
    "This message shares a sports opinion in an affirming and supportive way. It does not target or insult anyone.",
    "This message talks about TV shows in a light and personal way. It does not include stereotypes or aggressive tone.",
    "This message expresses political skepticism while staying polite. It does not include threats, hate, or misinformation.",
    "This message shows support and wishes someone good health. It is friendly and caring.",
    "This message is a sarcastic political opinion. It expresses light criticism but remains respectful. It includes irony, but no harm.",
    "This message discusses something funny and playfully suggests writing a book about it. It is not offensive.",
]

def build_prompt(setup, batch, mode="standard"):
    """
    Construct a prompt for the LLM with optional few-shot examples and specific reasoning mode.

    Args:
        setup: Few-shot setup (P1–P4)
        batch: List of tweets
        mode: Reasoning mode ('standard', 'targeted', 'IHC_labels')
    """
    n = len(batch)
    assert n > 0, "Batch cannot be empty."
    assert mode in {"standard", "targeted", "IHC_labels"}, "Mode must be one of: standard, targeted, IHC_labels"

    numbered_tweets = "\n".join([f"{j+1}. {t}" for j, t in enumerate(batch)])

    if mode == "standard":
        base = """Classify the following texts into one of the following categories:
- abusive (ABU)
- not abusive (NOTABU)

Instructions for each text:
1. Write 1–2 short sentences explaining why the text is classified as ABU or NOTABU.
2. Then output the label on a separate line starting with: Label: ABU or Label: NOTABU.

Follow this example format:
Reasoning: <reasoning>
Label: <ABU or NOTABU>
"""
    elif mode == "targeted":
        base = """Classify the following texts into one of the following categories:
- abusive (ABU)
- not abusive (NOTABU)

Instructions for each text:
1. Write 1–2 short sentences explaining why the text is classified as ABU or NOTABU. Consider in your explanation if the text is targeted or untargeted.
2. Then output the label on a separate line starting with: Label: ABU or Label: NOTABU.

Follow this example format:
Reasoning: <reasoning>
Label: <ABU or NOTABU>
"""
    else:  # mode == "IHC_labels"
        base = """Classify the following texts into one of the following categories:
- abusive (ABU)
- not abusive (NOTABU)

Instructions for each text:
1. Write 1–2 short sentences explaining why the text is classified as ABU or NOTABU. Consider in your explanation if the text includes:
- white grievance
- incitement to violence
- inferiority language
- irony
- stereotypes and misinformation
- threatening and intimidation
2. Then output the label on a separate line starting with: Label: ABU or Label: NOTABU.

Follow this example format:
Reasoning: <reasoning>
Label: <ABU or NOTABU>
"""

    # Determine few-shot count
    shots = {
        "P1": 0,
        "P2": 1,
        "P3": 8,
        "P4": 12,
    }.get(setup, 0)

    # Split reasonings by class
    REASONING_BASE_ABU = REASONING_BASE[:12]
    REASONING_BASE_NOTABU = REASONING_BASE[12:]
    REASONING_TARGET_ABU = REASONING_TARGET[:12]
    REASONING_TARGET_NOTABU = REASONING_TARGET[12:]
    REASONING_IHC_ABU = REASONING_IHC[:12]
    REASONING_IHC_NOTABU = REASONING_IHC[12:]

    if shots > 0:
        abu_examples = [ex for ex in EXAMPLES_ALL if ex[0] == "ABU"][:shots]
        notabu_examples = [ex for ex in EXAMPLES_ALL if ex[0] == "NOTABU"][:shots]

        if len(abu_examples) < shots or len(notabu_examples) < shots:
            raise ValueError("Not enough ABU or NOTABU examples available.")

        example_lines = []

        for i in range(shots):
            label, text = abu_examples[i]
            if mode == "standard":
                reasoning = REASONING_BASE_ABU[i]
            elif mode == "targeted":
                reasoning = REASONING_TARGET_ABU[i]
            else:
                reasoning = REASONING_IHC_ABU[i]
            example_lines.append(f"Text: {text}\nReasoning: {reasoning}\nLabel: {label}")

        for i in range(shots):
            label, text = notabu_examples[i]
            if mode == "standard":
                reasoning = REASONING_BASE_NOTABU[i]
            elif mode == "targeted":
                reasoning = REASONING_TARGET_NOTABU[i]
            else:
                reasoning = REASONING_IHC_NOTABU[i]
            example_lines.append(f"Text: {text}\nReasoning: {reasoning}\nLabel: {label}")

        base += "\nHere are a few examples of labeled texts:\n" + "\n\n".join(example_lines) + "\n"

    base += f"""
Now classify the following texts. ALWAYS output the instruction steps for EACH of the {n} texts.

Texts:
{numbered_tweets}
"""

    return base

def extract_labels_fallback(text, expected_count):
    """
    Fallback label extractor for chain-of-thought output format.
    Searches for lines starting with 'Label: <LABEL>'.
    """
    labels = []
    pattern = re.compile(r"Label\s*[:\-]?\s*(ABU|NOTABU)", re.IGNORECASE)

    for line in text.strip().splitlines():
        match = pattern.search(line)
        if match:
            label = match.group(1).upper()
            labels.append(label)

    return labels if len(labels) == expected_count else None

def main():
    """
    Run CoT prompting for binary abuse classification with reasoning.
    Saves prediction results and classification report.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", choices=["P1", "P2", "P3", "P4"], required=True, help="Setup (P1–P4)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for the model (default: 0.0)")
    parser.add_argument("--mode", choices=["standard", "targeted", "IHC_labels"], default="standard", help="Prompt mode")
    args = parser.parse_args()
    setup = args.setup
    temperature = args.temperature

    # Load test file
    df = pd.read_csv("../data/test/OLID_AbuseEval_test.tsv", sep="\t")
    tweets = df["tweet"].tolist()
    print(f"▶ Running binary setup {setup} on {len(tweets)} tweets...")

    predictions = []
    batch_size = 3  # Use small batches for CoT

    for i in range(0, len(tweets), batch_size):
        batch = tweets[i:i+batch_size]
        prompt = build_prompt(setup, batch, mode=args.mode)

        try:
            # Send prompt to local LLM and parse prediction labels
            result = query_LLM(client, prompt, temp=temperature)
            print(f"\n=== MODEL OUTPUT ({i}) ===\n{result}\n")
            labels = extract_labels_fallback(result, len(batch))
            if labels:
                predictions.extend(labels)
            else:
                print(f"⚠️ Unable to parse labels from output:\n{result}")
                predictions.extend(["ERROR"] * len(batch))
        except Exception as e:
            print(f"⚠️ Error on batch {i}: {e}")
            predictions.extend(["ERROR"] * len(batch))

    # Attach predictions to original DataFrame
    df = df.iloc[:len(predictions)]
    df["prediction"] = predictions

    # Save predictions
    os.makedirs("../predictions", exist_ok=True)
    pred_file = f"../predictions/binary-CoT-{args.mode}-{setup}-t{temperature:.2f}_predictions.csv"
    report_file = f"../predictions/binary-CoT-{args.mode}-{setup}-t{temperature:.2f}_report.txt"
    df.to_csv(pred_file, index=False)
    print(f"✅ Predictions saved to {pred_file}")

    # Generate classification report if gold labels exist
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
            with open(report_file, "w") as f:
                f.write(report)
            print(f"📊 Report saved to {report_file}")
    else:
        print("⚠️ Column 'abuse' not found for evaluation.")

if __name__ == "__main__":
    main()
