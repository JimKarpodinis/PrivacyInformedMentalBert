import argparse
from utils import load_data, recast_columns, split_dataset
from datasets import Dataset, DatasetDict


def select_columns(dataset: Dataset) -> Dataset:

    return dataset.select_columns(["text", "label"]) 

if __name__ == "__main__":

    parser = argparse.ArgumentParser() 

    parser.add_argument("--data_dir", help="The directory where the data reside", type=str)
    
    args = parser.parse_args()
    data_dir = args.data_dir

    dataset = load_data(data_dir)
    dataset = recast_columns(dataset, ["0", "1"])
    
    dataset = select_columns(dataset)
    split_dataset(dataset)
