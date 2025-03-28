"""
Validate the performance of the models which was stated in the MentalBert paper.
The models to be checked are BERT, RoBERTa, BioBERT, ClinicalBERT, MentalBERT, MentalRoBERTa
"""

from trasnformers import PretrainedModel
import evaluate
from datasets import Dataset


def check_model_performance(model: PretrainedModel, dataset: Dataset) -> dict:
    
    labels = dataset.select_columns(["label"])
    features = dataset.select_columns(["text"])
    

    logits = model(dataset)
    
    eval_preds = (logits, labels)
    return compute_metrics(eval_preds)


def compute_metrics(eval_preds):

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")


    logits, labels = eval_preds
    preds = np.argmax(logits, axis=1)
    

    accuracy = accuracy_metric.compute(preds, labels)
    f1 = f1_metric.compute(preds, labels)   


    return {"accuracy": accuracy, "f1": f1}
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", help=" The hf model name", type=str)

    args = parser.parse_args()
    model_name = args.model_name

    model = PretrainedModel(model_name)

    check_model_performance(model)
