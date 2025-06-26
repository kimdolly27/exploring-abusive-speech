"""
Train and evaluate BERT and HateBERT on the AbuseEval (+ HIC) dataset.

Supports binary and ternary abuse classification.
Saves predictions, classification reports, and confusion matrix plots.
"""

import os
import shutil
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed
)
from datasets import Dataset, DatasetDict, ClassLabel, Features, Value

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics from model predictions.

    Args:
        eval_pred: tuple of (logits, true_labels)

    Returns:
        dict with accuracy and weighted F1 score
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    from sklearn.metrics import accuracy_score, f1_score
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }

def tokenize_and_align_labels(examples, tokenizer, max_seq_length):
    """
    Tokenize tweets and assign labels for model input.

    Args:
        examples: batch of examples from dataset
        tokenizer: HuggingFace tokenizer instance
        max_seq_length: maximum token length for truncation

    Returns:
        dict with tokenized inputs and corresponding labels
    """
    tokenized_inputs = tokenizer(
        examples["tweet"],
        truncation=True,
        padding="max_length",
        max_length=max_seq_length
    )
    tokenized_inputs["labels"] = examples["label"]
    return tokenized_inputs

def run_training(model_checkpoint, train_file, test_file, label_mode="binary", max_seq_length=128, batch_size=32, num_epochs=5):
    """
    Train a transformer-based classifier on abuse data.

    Args:
        model_checkpoint: model name or path (e.g. 'bert-base-uncased')
        train_file: path to training TSV file
        test_file: path to test TSV file
        label_mode: 'binary' or 'ternary'
        max_seq_length: token truncation length
        batch_size: training batch size
        num_epochs: number of training epochs

    Returns:
        model_dir: directory where the trained model is saved
    """
    df_train_full = pd.read_csv(train_file, sep='\t')[['id', 'tweet', 'abuse']]
    df_test = pd.read_csv(test_file, sep='\t')[['id', 'tweet', 'abuse']]

    if label_mode == "binary":
        df_train_full['abuse'] = df_train_full['abuse'].replace({'IMP': 'ABU', 'EXP': 'ABU'})
        df_test['abuse'] = df_test['abuse'].replace({'IMP': 'ABU', 'EXP': 'ABU'})
        label_names = ['NOTABU', 'ABU']
    else:
        label_names = ['NOTABU', 'IMP', 'EXP']

    train_df, val_df = train_test_split(df_train_full, test_size=0.1, stratify=df_train_full['abuse'], random_state=42)

    dataset_train = Dataset.from_pandas(train_df, preserve_index=False)
    dataset_val = Dataset.from_pandas(val_df, preserve_index=False)
    dataset_test = Dataset.from_pandas(df_test, preserve_index=False)

    features = Features({
        'id': Value('string'),
        'tweet': Value('string'),
        'abuse': ClassLabel(names=label_names),
    })

    dataset_train = dataset_train.cast(features)
    dataset_val = dataset_val.cast(features)
    dataset_test = dataset_test.cast(features)

    dataset = DatasetDict({
        'train': dataset_train,
        'validation': dataset_val,
        'test': dataset_test
    })

    dataset = dataset.rename_column("abuse", "label")
    dataset = dataset.cast_column("label", ClassLabel(names=label_names))

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenized_data = dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer, max_seq_length), batched=True)

    label_list = dataset["train"]["label"]
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(label_list), y=label_list)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(label_names),
        id2label={i: l for i, l in enumerate(label_names)},
        label2id={l: i for i, l in enumerate(label_names)}
    )

    dataset_name = os.path.splitext(os.path.basename(train_file))[0]
    model_dir = f"./model/{model_checkpoint.replace('/', '_')}_{dataset_name}_{label_mode}"
    os.makedirs(model_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=model_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        logging_steps=50,
        save_total_limit=1,
        save_strategy="epoch",
        seed=42,
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )

    class WeightedTrainer(Trainer):
        """
        Trainer subclass with support for class-weighted loss.
        """
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor.to(logits.device))
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["validation"],
        data_collator=DataCollatorWithPadding(tokenizer),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print(f"\n=== Training model: {model_checkpoint} on dataset: {dataset_name} as {label_mode} classification ===")
    trainer.train()

    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    test_metrics = trainer.evaluate(tokenized_data["test"])
    print(f"=== Final Test Set Performance: {test_metrics} ===\n")

    return model_dir

def evaluate_model(model_dir, test_file, output_dir="./predictions"):
    """
    Evaluate a trained model on the test set and save results.

    Args:
        model_dir: path to trained model directory
        test_file: path to test TSV file
        output_dir: folder to store predictions and reports
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    df_test = pd.read_csv(test_file, sep='\t')[['id', 'tweet', 'abuse']]
    model_name = os.path.basename(model_dir)

    print(f"\n=== Evaluating model: {model_name} ===")

    if "binary" in model_name:
        label_names = ["ABU", "NOTABU"]
        df_eval = df_test.copy()
        df_eval["abuse"] = df_eval["abuse"].replace({"IMP": "ABU", "EXP": "ABU"})
    elif "ternary" in model_name:
        label_names = ["EXP", "IMP", "NOTABU"]
        df_eval = df_test.copy()
    else:
        print(f"Skipping {model_name} (label mode not found in name)")
        return

    label2id = {label: idx for idx, label in enumerate(label_names)}
    id2label = {idx: label for label, idx in label2id.items()}

    df_eval = df_eval[df_eval["abuse"].isin(label2id.keys())]
    df_eval["label"] = df_eval["abuse"].map(label2id)
    dataset_eval = Dataset.from_pandas(df_eval[['id', 'tweet', 'label']], preserve_index=False)

    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def tokenize(example):
        return tokenizer(example["tweet"], truncation=True, padding="max_length", max_length=128)

    tokenized_eval = dataset_eval.map(tokenize, batched=True)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./tmp_eval_output",
            per_device_eval_batch_size=32,
            no_cuda=(device.type == "cpu"),
            logging_dir="./tmp_eval_logs"
        ),
        tokenizer=tokenizer
    )

    predictions, label_ids, _ = trainer.predict(tokenized_eval)
    predicted_indices = np.argmax(predictions, axis=1)

    results_df = pd.DataFrame({
        "id": df_eval["id"],
        "tweet": df_eval["tweet"],
        "true_label": df_eval["abuse"],
        "predicted_label": [id2label[i] for i in predicted_indices]
    })

    results_df.to_csv(f"{output_dir}/predictions_{model_name}.csv", index=False)

    report_txt = classification_report(
        results_df["true_label"],
        results_df["predicted_label"],
        labels=label_names,
        target_names=label_names
    )
    with open(f"{output_dir}/classification_report_{model_name}.txt", "w") as f:
        f.write(report_txt)
    print(report_txt)

    cm = confusion_matrix(results_df["true_label"], results_df["predicted_label"], labels=label_names)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix_{model_name}.png")
    plt.close()

    shutil.rmtree("./tmp_eval_output", ignore_errors=True)
    shutil.rmtree("./tmp_eval_logs", ignore_errors=True)

def main():
    """
    Run all combinations of model + dataset + label_mode for training and evaluation.
    """
    test_file = "../data/test/olid_abuseeval_test.tsv"
    combinations = [
        ("GroNLP/hateBERT", "../data/train/olid_abuseeval_train.tsv"),
        ("GroNLP/hateBERT", "../data/train/olid_abuseeval_hic_train.tsv"),
        ("bert-base-uncased", "../data/train/olid_abuseeval_train.tsv"),
        ("bert-base-uncased", "../data/train/olid_abuseeval_hic_train.tsv"),
    ]
    label_modes = ["binary", "ternary"]

    for model_checkpoint, train_file in combinations:
        for label_mode in label_modes:
            model_dir = run_training(
                model_checkpoint=model_checkpoint,
                train_file=train_file,
                test_file=test_file,
                label_mode=label_mode
            )
            evaluate_model(
                model_dir=model_dir,
                test_file=test_file,
                output_dir="./predictions"
            )

if __name__ == "__main__":
    main()