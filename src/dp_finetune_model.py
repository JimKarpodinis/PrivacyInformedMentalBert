"""
Finetune the models with diffential privacy contraints
The models to be checked are BERT, RoBERTa, BioBERT, ClinicalBERT, MentalBERT, MentalRoBERTa
"""
from datetime import datetime
import argparse
import os
import json
from datasets import load_dataset
from opacus import PrivacyEngine
import datasets
from transformers import BertForSequenceClassification, AutoTokenizer, BertConfig
import torch
from utils import tokenize_sentences
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW


def define_training_args(
        file_name: str="training_hyperparams.json") -> dict:
    
    file_name = os.path.join("json", file_name)

    with open(file_name, "rb") as f:

        configuration = json.load(f)

        training_args = configuration

    return training_args


def set_seed(seed: int = 42):
    random.seed(seed)                         # Python random
    np.random.seed(seed)                      # NumPy
    torch.manual_seed(seed)                   # PyTorch (CPU)
    torch.cuda.manual_seed(seed)              # PyTorch (GPU)
    torch.cuda.manual_seed_all(seed)          # If multiple GPUs

    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model, only_trainable=True):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
        # Only clf params are counted, expect small number
    else:
        return sum(p.numel() for p in model.parameters())


def train_one_epoch(epoch_index: int, tb_writer: SummaryWriter,
        model: BertForSequenceClassification, train_loader: DataLoader,
        optimizer) -> torch.Tensor:

    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, inputs in enumerate(train_loader):
        # Every data instance is an input + label pair

        del inputs["text"]

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(**inputs)

        # Compute the loss and its gradients
        loss = outputs.loss
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss / (i + 1)  # loss per batch
            print('batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


def main(model_name: str, data_dir: str, epochs: int, seed: int):

    # TODO: Modify inference to inference per seed
    # TODO: Use multiple seeds and find avg and sd
    #  Might need to test directly after training
    # TODO: Get last epoch epsilon
    # TODO: Check when does it basically converge
    # TODO: Create inference statistics file
    # Dir structure to be like 'seed=<seed>'
    # TODO: Run training without dp
    # TODO: Run torch summary 


    set_seed(seed)
    hf_token = os.getenv("HF_TOKEN")
    privacy_engine = PrivacyEngine()
    #------------------------------------------------------------------------
    dataset_dict = load_dataset("csv", data_dir=data_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dataset_dict = dataset_dict.map(
            lambda batch: tokenize_sentences(batch,
                tokenizer), batched=True)

    dataset_dict.set_format("torch", device=device)

    training_args = define_training_args()

    train_dataset = dataset_dict["train"]


    validation_dataset = dataset_dict["validation"]

    # config = BertConfig(max_position_embeddings=10000)

    model = BertForSequenceClassification.from_pretrained(
            model_name, token=hf_token) # , config=config)

    model.to(device)

    for param in model.base_model.parameters():
        param.requires_grad = False

    optimizer = AdamW(model.classifier.parameters())

    train_loader = DataLoader(train_dataset,
            batch_size=training_args["per_device_train_batch_size"],
            shuffle=True) # , collate_fn=data_collator)

    validation_loader = DataLoader(validation_dataset,
            batch_size=training_args["per_device_eval_batch_size"],
            shuffle=True) # , collate_fn=data_collator)

    #----------------------------------------------------------------------------

    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(training_args["logging_dir"])

    EPOCHS = epochs

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)

        model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=1.1,
        max_grad_norm=1.0,
        )

        avg_loss = train_one_epoch(epoch, writer, model, train_loader, optimizer)

        epsilon, best_alpha = privacy_engine.get_epsilon(delta=10**-5)

        writer.add_scalars('Epsilon per Epoch', {"Epsilon": epsilon}, epoch + 1)

        writer.add_scalars('Random Seed', {"Seed": seed}, epoch + 1)

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vinputs in enumerate(validation_loader):

                del vinputs["text"]

                voutputs = model(**vinputs)
                vloss = voutputs.loss
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:

            best_epsilon, _ = privacy_engine.get_epsilon(
                    delta=10**-5)

            best_model = model


    best_model._module.state_dict(["opacus_epsilon"]) = best_epsilon

    model_path = f"{training_args['output_dir']}/seed={seed}"
    os.makedirs(model_path, exist_ok=True)

    best_model._module.save_pretrained(model_path, from_pt=True)
    print(f"(ε = {epsilon:.2f}, δ = {delta}) for α = {best_alpha}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", help="The hf model name", type=str)

    parser.add_argument("--data_dir",
            help="The directory where the data reside", type=str)

    parser.add_argument("--epochs", help="The number of epochs to train the model", type=int)

    parser.add_argument("--seed", help="The random seed number to be used", type=int)

    args = vars(parser.parse_args())
    

    main(**args)
