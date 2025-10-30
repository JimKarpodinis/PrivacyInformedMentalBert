import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, BertForSequenceClassification, AutoTokenizer
from datasets import load_dataset, Dataset
import evaluate
from sklearn.metrics import classification_report
from utils import tokenize_sentences, define_training_args, read_json_file, write_json_file


def main(data_dir: str):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    training_args = define_training_args()

    test_dataset  = load_dataset("csv", data_dir=data_dir, split="test")
    test_dataset.set_format("torch", device=device)

    tokenizer = AutoTokenizer.from_pretrained("mental/mental-bert-base-uncased")

    test_dataset = test_dataset.map(lambda batch: tokenize_sentences(
        batch, tokenizer=tokenizer), batched=True)

    test_loader = DataLoader(test_dataset, batch_size=64)

    compute_eval_statistics(training_args.output_dir, test_loader, device) 


def compute_eval_statistics(model_dir: str, dataloader: DataLoader, device: str):

    seeds_used = [1, 1234, 100, 42, 0]

    evaluation_metrics = []
    for seed in seeds_used:

        seed_model_dir = os.path.join(model_dir, f"seed={seed}")

        model = _load_model(seed_model_dir)
        model.to(device)

        seed_evaluation_metrics, seed_clf_report = evaluate_model(
            model, dataloader)

        if seed == 42: 
            final_clf_report = seed_clf_report 

        evaluation_metrics.append(seed_evaluation_metrics)


    final_evaluation_metrics = {"f1": torch.mean(torch.tensor([metrics["f1"] for metrics in evaluation_metrics]), dim=0).item(), 
                            "recall": torch.mean(torch.tensor([metrics["recall"] for metrics in evaluation_metrics]), dim=0).item(),
                            "precision": torch.mean(torch.tensor([metrics["precision"] for metrics in evaluation_metrics]), dim=0).item(),
                            "accuracy": torch.mean(torch.tensor([metrics["accuracy"] for metrics in evaluation_metrics]), dim=0).item()}

    

    final_clf_report_path = os.path.join(model_dir, "clf_report.json")
    final_evals_path = os.path.join(model_dir, "test_results.json")

    write_json_file(final_evals_path, final_evaluation_metrics)
    write_json_file(final_clf_report_path, final_clf_report)


def _load_model(model_dir):

    return BertForSequenceClassification.from_pretrained(model_dir)


def evaluate_model(model: AutoModel, dataloader: DataLoader) -> dict:

    f1_metric = evaluate.load("f1")
    recall_metric = evaluate.load("recall")
    precision_metric = evaluate.load("precision")
    accuracy_metric = evaluate.load("accuracy")

    preds_tot = []
    labels_tot = []

    for batch in dataloader:

        del batch["text"]

        logits = model(**batch).logits
       
        preds = logits.argmax(dim=1)

        labels = batch["labels"]

        preds_tot.extend(preds.tolist())
        labels_tot.extend(labels.tolist())


    num_labels = logits.shape[1]

    averaging_method = "macro" if num_labels > 2 else "binary"

    f1 = f1_metric.compute(
            references=labels_tot, predictions=preds_tot,
            average=averaging_method)["f1"]
    
    recall = recall_metric.compute(
            references=labels_tot, predictions=preds_tot,
            average=averaging_method)["recall"]

    precision = precision_metric.compute(
            references=labels_tot, predictions=preds_tot,
            average=averaging_method)["precision"]

    accuracy = accuracy_metric.compute(
            references=labels_tot, predictions=preds_tot,
           )["accuracy"]

    evaluation_metrics = {"f1": f1,
            "recall": recall, "precision": precision, "accuracy": accuracy}

    clf_report = classification_report(labels_tot, preds_tot, output_dict=True)

    return evaluation_metrics, clf_report
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", help="The directory where the data reside", type=str)

    args = vars(parser.parse_args())
    main(**args)


