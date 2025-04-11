import os
from utils import load_data, split_dataset, write_dataset, recast_columns
import argparse
from datasets import Dataset, DatasetDict
from datasets.exceptions import DatasetGenerationCastError


def extract_label_from_response(dataset: Dataset) -> Dataset:
    """
    Get the depression label from the first sentence of the respone field. 

    This field contains the desired generated response of the model.
    """

    return dataset.map(lambda record: {"label": (
        1 if "yes" == record["response"].split(".")[0] 
        else 0)})


def rename_column(dataset: Dataset, previous_col_name: str="post", next_col_name: str="text") -> Dataset:

    return dataset.rename_column(previous_col_name, next_col_name)


def _extract_question(record: dict) -> dict:
    """Extract question from test dataset"""

    question_index = record["query"].find("Question")
    
    record["question"] = record["query"][question_index:]

    return record


def _extract_post(record: dict) -> dict:
    """Extract post from test dataset"""

    question_index = record["query"].find("Question")
    
    record["post"] = record["query"][:question_index]

    return record


def _remove_post_prefix(record: dict) -> dict:

    """Remove the prefix 'Post:' from the post column"""

    post_index = record["post"].find(":")
    # Find the first instance of the colon character
    # It is always after the word post. 

    record["post"] = record["post"][post_index:]

    return record


def extract_post(dataset: Dataset) -> Dataset:

    return dataset.map(lambda record: _extract_post(record))


def extract_question(dataset: Dataset) -> Dataset:

    return dataset.map(lambda record: _extract_question(record),
            remove_columns=["query"])


def remove_post_prefix(record: dict) -> dict:
    
    return dataset.map(lambda record: _remove_post_prefix(record))


def select_columns(dataset: Dataset) -> Dataset:

    return dataset.select_columns(["text", "label"]) 


if __name__ == "__main__":

    parser = argparse.ArgumentParser() 

    parser.add_argument("--data_dir", help="The directory where the data reside", type=str)
    
    args = parser.parse_args()
    data_dir = args.data_dir

    try:
        dataset = load_data(data_dir)

    except DatasetGenerationCastError: 
        test_data_path = os.path.join(data_dir, "DR_test.csv")

        dataset = load_data(data_files=test_data_path)
        # Must change the test split column names first

        dataset = extract_post(dataset)
        dataset = extract_question(dataset)
        dataset = rename_column(dataset, "gpt-3.5-turbo", "response")
        write_dataset(dataset, test_data_path)

        dataset = load_data(data_dir=data_dir)


    breakpoint()
    dataset = remove_post_prefix(dataset)

    dataset = rename_column(dataset)
    dataset = extract_label_from_response(dataset)
    dataset = select_columns(dataset)
    
    label_names = ["0", "1"]
    dataset = recast_columns(dataset, label_names)
    split_dataset(dataset, data_dir)

