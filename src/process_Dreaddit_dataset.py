from utils import load_data
from datasets import Dataset, DatasetDict


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

    breakpoint()
    dataset = load_data(data_dir)
    
    dataset = select_columns(dataset)
    split_dataset(dataset)
