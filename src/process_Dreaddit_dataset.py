import argparse
from utils import load_data, recast_columns, split_dataset, rename_columns
from datasets import Dataset, DatasetDict


def select_columns(dataset: Dataset) -> Dataset:

    return dataset.select_columns(["text", "label"]) 


if __name__ == "__main__":

    parser = argparse.ArgumentParser() 

    parser.add_argument("--data_dir", help="The directory where the data reside", type=str)
    
    args = parser.parse_args()
    data_dir = args.data_dir

    dataset = load_data(data_dir, split="all")
    
    dataset = select_columns(dataset)
    dataset = rename_columns(dataset)

    dataset = recast_columns(dataset, ["0", "1"])
    split_dataset(dataset, data_dir)
