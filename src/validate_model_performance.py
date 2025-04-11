"""
Validate the performance of the models which was stated in the MentalBert paper.
The models to be checked are BERT, RoBERTa, BioBERT, ClinicalBERT, MentalBERT, MentalRoBERTa
"""
import os
import argparse
import json

from transformers import AutoTokenizer, BertForSequenceClassification, TrainingArguments, DataCollatorWithPadding, Trainer
import evaluate
from datasets import Dataset
import torch
import numpy as np

from utils import load_data


def compute_metrics(eval_preds: tuple) -> dict:

    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    logits, labels = eval_preds
    preds = logits.argmax(axis=-1)
    breakpoint()

    recall = recall_metric.compute(predictions=preds, references=labels)
    f1 = f1_metric.compute(predictions=preds, references=labels)   

    return {"recall": recall, "f1": f1}
    

def define_trainer(model: BertForSequenceClassification,
        dataset: Dataset,
        collator: DataCollatorWithPadding,
        training_args: dict) -> Trainer:


    return Trainer(model=model,
            data_collator=collator,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset = dataset["train"],
            eval_dataset = dataset["validation"],)


def define_training_args(
        file_name: str="training_hyperparams.json") -> TrainingArguments:
    
    file_name = os.path.join("json", file_name)

    with open(file_name, "rb") as f:

        configuration = json.load(f)

        training_args = TrainingArguments(**configuration)

    return training_args


def tokenize_sentences(examples: list) -> list:

    return tokenizer(examples["text"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", help="The hf model name", type=str)

    parser.add_argument("--data_dir",
            help="The directory where the data reside", type=str)

    args = parser.parse_args()
    
    model_name = args.model_name
    data_dir = args.data_dir

    hf_token = os.getenv("HF_TOKEN")

    dataset = load_data(data_dir=data_dir)
    dataset.set_format("torch")

    model = BertForSequenceClassification.from_pretrained(model_name, token=hf_token)

    for param in model.base_model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = define_training_args()

    tokenized_dataset = dataset.map(tokenize_sentences, batched=True)

    trainer = define_trainer(model=model,
                training_args=training_args,
                collator=collator,
                dataset=tokenized_dataset)

    trainer.train()
    trainer.evaluate()
