Exploring Implicit Abusive Speech Detection: A Comprehensive Analysis of Fine-Tuning BERT and Prompting Qwen2.5

*** Abstract ***

Online content has become a significant part of our daily lives. This development made detecting abusive speech important for the overall well-being of society. One of the challenges in detecting abusive speech involves the nuanced and implicit ways in which it can be expressed. Since the early 2010s, researchers have made several attempts to tackle the detection of abusive forms of speech, with recent work showing promising results for transformer-based models and generative large language models (LLMs).

In this research, I aim to contribute to the detection of abusive speech by answering the question of whether prompt engineering offers advantages over a fine-tuned BERT model, particularly in identifying implicit cases of abusive speech. I conduct two main experiments — fine-tuning BERT-based models and prompting Qwen2.5 — across both binary (abusive vs. not abusive) and ternary (explicit abuse, implicit abuse, not abusive) classification tasks, and evaluate the performances on the AbuseEval test set. Finally, I conduct a thorough error analysis to examine how errors, and in particular mistakes in implicit abusive speech, affect the model’s results.

The results show that fine-tuning still delivered better overall performance, achieving a macro-averaged F1-score of 0.60, with 0.29 for implicit cases. The best-performing prompting strategy combined Chain-of-Thought (CoT) with considering targetness, reaching a macro-averaged F1-score of 0.52, with 0.25 in the implicit class. The error analysis revealed that helping the model understand the boundary between explicit and implicit abuse, and implicit and non-abusive, through improving understanding of the target and the context within tweets, is key in reducing misclassification in abusive speech, particularly in implicit cases.

*** Author ***
K.D. Gerritsen
June 27, 2025
https://github.com/kimdolly27

*** Directory structure ***
exploring__abusive_speech_detection/
├── data/
├── error_analysis/
├── fine-tuning_BERT/
│ └── train_BERT.py
├── model/
├── predictions/
├── prompting_experiments/
│ ├── prompt-binary-base.py
│ ├── prompt-binary-CoT.py
│ ├── prompt-ternary-base.py
│ └── prompt-ternary-CoT.py
├── analyse_errors.py
├── compute_statistics.py
├── merge_datasets.py
├── README.txt
└── requirements.txt

*** Data ***
NOTE: Make sure the input files exist at the specified paths before running this script.
Expected folders:
- OLID training/test files: ./data/OLIDv1/ - https://sites.google.com/site/offensevalsharedtask/olid
- AbuseEval labels: ./data/AbuseEval-master/ - https://github.com/tommasoc80/AbuseEval
- HIC posts: ./data/implicit-hate-corpus/ - https://github.com/SALT-NLP/implicit-hate

*** Model ***
Note: This prompting_experiments script expects a locally running language model server with an OpenAI-compatible API at http://localhost:8000/v1.

Model can be downloaded via: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF

Example files:
- qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf
- qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf

*** Scripts ***

train_BERT.py - Train and evaluate BERT and HateBERT on the AbuseEval (+ HIC) dataset. Supports binary and ternary abuse classification.

prompting_experiments/
These scripts classify tweets as abusive or not using a local LLaMA model with one-shot and few-shot prompting. Each script supports different prompting styles and saves the model’s prediction, and supports evaluation. Key Features:
- One-shot and few-shot classification (P1–P4 setups)
- Optional reasoning-based (CoT) prompting
- Multiple prompt strategies
- Saves predictions and can run evaluations

prompt-binary-base.py
    Binary classification (ABU / NOTABU) using basic prompting.
    Modes: base, def
prompt-binary-CoT.py
    Binary classification (ABU / NOTABU) using reasoning (chain-of-thought).
    Modes: standard, targeted, IHC_labels
prompt-ternary-base.py
    Ternary classification (EXP / IMP / NOTABU) using basic prompting.
    Modes: base, def
prompt-ternary-CoT.py
    Ternary classification (EXP / IMP / NOTABU) using reasoning (chain-of-thought).
    Modes: standard, targeted, IHC_labels

** Utility Scripts **
analyse_errors.py - Script for error analysis and visualization of ternary classification predictions.
compute_statistics.py - Loads and analyzes the AbuseEval and HIC datasets, printing abuse class distributions and subtask label counts.
merge_datasets.py - Prepare training and test datasets by merging OLID, AbuseEval, and HIC datasets.

*** requirements.txt ***
All required Python packages are listed in requirements.txt.
Install them with:  
    pip install -r requirements.txt
