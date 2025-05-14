"""
Validate the performance of the models which was stated in the MentalBert paper.
The models to be checked are BERT, RoBERTa, BioBERT, ClinicalBERT, MentalBERT, MentalRoBERTa
"""
import os
import argparse
import json
from typing import Type, Callable

from transformers import AutoTokenizer, BertForSequenceClassification, TrainingArguments, DataCollatorWithPadding, Trainer
import evaluate
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
import numpy as np

from utils import load_data, define_training_args


def compute_metrics(eval_preds: tuple) -> dict:

    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    logits, labels = eval_preds
    preds = logits.argmax(axis=-1)

    num_labels = logits.shape[1]
    averaging_method = "weighted" if num_labels > 2 else "binary"

    recall = recall_metric.compute(
            predictions=preds, references=labels,
            average=averaging_method)["recall"]

    f1 = f1_metric.compute(
            predictions=preds,
            references=labels, average=averaging_method)["f1"]   

    return {"recall": recall, "f1": f1}
    

def define_trainer(model: BertForSequenceClassification,
        dataset: Dataset,
        collator: DataCollatorWithPadding,
        training_args: dict,
        trainer_cls: Type[Trainer]) -> Trainer:

    return trainer_cls(model=model,
            data_collator=collator,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset = dataset["train"],
            eval_dataset = dataset["validation"],)


def tokenize_sentences(examples: list, tokenizer: AutoTokenizer) -> list:

    return tokenizer(examples["text"])


def main(model_name: str, data_dir: str, trainer_cls: Type[Trainer]):

    hf_token = os.getenv("HF_TOKEN")

    #----------------------------------------------------------------------------
    dataset = load_data(data_dir=data_dir)
    dataset.set_format("torch")

    num_labels = len(dataset["train"]["labels"].unique())

    model = BertForSequenceClassification.from_pretrained(model_name, token=hf_token, num_labels=num_labels)

    for param in model.base_model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    training_args = define_training_args()
    #----------------------------------------------------------------------------

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized_dataset = dataset.map(lambda batch: tokenize_sentences(batch, tokenizer=tokenizer), batched=True)

    trainer = define_trainer(model=model,
                training_args=training_args,
                collator=collator,
                dataset=tokenized_dataset,
                trainer_cls = trainer_cls)

    trainer.train()
    _, _, metrics = trainer.predict(tokenized_dataset["test"])
    trainer.save_metrics(split="test", metrics=metrics, combined=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", help="The hf model name", type=str)

    parser.add_argument("--data_dir",
            help="The directory where the data reside", type=str)

    args = parser.parse_args()
    
    model_name = args.model_name
    data_dir = args.data_dir

    main(trainer_cls= Trainer, model_name= model_name, data_dir= data_dir)


