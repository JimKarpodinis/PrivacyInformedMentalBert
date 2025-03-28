from datasets import load_dataset


def load_data(data_dir: str) -> Dataset:

    data_dir = os.path.join(os.getcwd(), data_dir)
    dataset = load_dataset("csv", data_dir=data_dir, split="train")
    return dataset


