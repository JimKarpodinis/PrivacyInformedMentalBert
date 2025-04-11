import os
import argparse
from datasets import Dataset, DatasetDict
from utils import load_data, recast_columns, split_dataset

def rename_columns(dataset: Dataset) -> Dataset:

    dataset = dataset.rename_column("sentence", "text")
    dataset = dataset.rename_column("top_label", "label")

    return dataset
 

def select_columns(dataset: Dataset) -> Dataset:

    return dataset.select_columns(["text", "label"])


def filter_non_stressors(dataset: Dataset) -> Dataset:
    
    """Filter out records which signify that there is no stressor"""

    return dataset.filter(lambda record: record["is_stressor"] == 1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser() 

    parser.add_argument("--data_dir", help="The directory where the data reside", type=str)
    
    args = parser.parse_args()
    data_dir = args.data_dir

    dataset = load_data(data_dir, split="all")

    label_names = ["Emotional Turmoil", "Everyday Decision Making",
            "Other", "Work", "Social Relationships",
            "School", "Family Issues", "Financial Problem",
            "Health, Fatigue, or Physical Pain"]

    dataset = filter_non_stressors(dataset)
    dataset = rename_columns(dataset)
    dataset = select_columns(dataset)
    breakpoint()
    dataset = recast_columns(dataset, label_names)

    split_dataset(dataset, data_dir)
