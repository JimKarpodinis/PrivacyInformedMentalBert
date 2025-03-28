from utils import load_data
from datasets import Dataset, DatasetDict


def extract_label_from_response(dataset: Dataset) -> Dataset:
    """
    Get the depression label from the first sentence of the respone field. 

    This field contains the desired generated response of the model.
    """

    return dataset.map(lambda record: {"label": (
        1 if "yes" == record["response"].split(".")[0] 
        else 0)})


def rename_column(dataset: Dataset) -> Dataset:

    return dataset.rename_column("post", "text")


def _remove_post_prefix(record: dict) -> dict:

    """ Remove the prefix 'Post:' from the post column"

    post_index = record["post"].find(":")
    # Find the first instance of the colon character
    # It is always after the word post. 

    record["post"] = record["text"][post_index:]

    return record


def remove_post_prefix(record: dict) -> dict:
    
    return dataset.map(lambda record: _remove_post_field_prefix(record))


def select_columns(dataset: Dataset) -> Dataset

    return dataset.select_columns(["text", "label"]) 


def split_dataset(dataset: Dataset, file_name: str) -> Dataset:
    """Split dataset to training and testing """

    dataset_dict = dataset.train_test_split(test_size=0.2, stratify="labels")
    dataset_train_dict = dataset_dict['train'].train_test_split(test_size=0.2, stratify="labels")

    dataset_train = dataset_train_dict['train']
    dataset_validation = dataset_train_dict['test']
    dataset_test = dataset_dict['test']

    file_name = file_name.removesuffix(".csv")
    
    dataset_train.to_csv(f"./data/processed/Dreaddit_dataset/{file_name}_train_dataset.csv", index = False)
    dataset_validation.to_csv(f"./data/processed/Dreaddit_dataset/{file_name}_validation_dataset.csv", index = False)
    dataset_test.to_csv(f"./data/processed/Dreaddit_dataset/{file_name}_test_dataset.csv", index = False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser() 

    parser.add_argument("--data_dir", help="The directory where the data reside", type=str)
    
    args = parser.parse_args()
    data_dir = args.data_dir

    dataset = load_data(data_dir)
    dataset = remove_post_prefix(dataset)

    dataset = rename_column(dataset)
    dataset = extract_label_from_response(dataset)
    dataset = select_columns(dataset)
    
    split_dataset(dataset)
