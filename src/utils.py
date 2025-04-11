from datasets import load_dataset, Dataset, ClassLabel
import os
from typing import Union


def recast_columns(dataset: Dataset, class_names: list) -> Dataset:

    features = dataset.features.copy()

    features["label"] = ClassLabel(names=class_names)

    return dataset.cast(features)


def load_data(data_dir: Union[str, None] =  None,
        data_files: Union[str, None] = None, split: Union[str, None]= None) -> Dataset:


    error_msg = "Please provide either data directory or dataset path"

    assert (data_dir is not None) or (data_files is not None), error_msg

    if data_dir:

        data_dir = os.path.join(os.getcwd(), data_dir)
        dataset = load_dataset("csv", data_dir=data_dir, split=split)

    else:
        data_files = os.path.join(os.getcwd(), data_files)
        dataset = load_dataset("csv", data_files=data_files, split="train")


    return dataset


def split_dataset(dataset: Dataset, data_dir: str) -> Dataset:
    """Split dataset to training and testing """

    dataset_dict = dataset.train_test_split(test_size=0.2, stratify_by_column="label")
    dataset_train_dict = dataset_dict['train'].train_test_split(test_size=0.2, stratify_by_column="label")

    dataset_train = dataset_train_dict['train']
    dataset_validation = dataset_train_dict['test']
    dataset_test = dataset_dict['test']

    file_name = os.path.basename(data_dir)
    
    train_dir = f"data/processed/{file_name}/train_dataset.csv"
    validation_dir = f"data/processed/{file_name}/validation_dataset.csv"
    test_dir = f"data/processed/{file_name}/test_dataset.csv"

    train_dir = os.path.join(os.getcwd(), train_dir)
    test_dir = os.path.join(os.getcwd(), test_dir)
    validation_dir = os.path.join(os.getcwd(), validation_dir)

    write_dataset(dataset_train, train_dir)
    write_dataset(dataset_validation, validation_dir)
    write_dataset(dataset_test, test_dir)
    

def write_dataset(dataset: Dataset, data_dir: str):

    dataset.to_csv(data_dir, index = False)
