"""
Finetune the models with diffential privacy contraints
The models to be checked are BERT, RoBERTa, BioBERT, ClinicalBERT, MentalBERT, MentalRoBERTa
"""
from datetime import datetime
import random
import argparse
import os
# from opacus.utils.module_utils import convert_grad_sample_modules
import json
from datasets import load_dataset
from opacus import PrivacyEngine
import numpy as np
import datasets
from transformers import BertForSequenceClassification, AutoTokenizer, BertConfig
from torch.optim.lr_scheduler import LinearLR
import torch
from utils import tokenize_sentences, define_training_args
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, AdamW


def set_seed(seed: int = 42):
    random.seed(seed)                         # Python random
    np.random.seed(seed)                      # NumPy
    torch.manual_seed(seed)                   # PyTorch (CPU)
    torch.cuda.manual_seed(seed)              # PyTorch (GPU)
    torch.cuda.manual_seed_all(seed)          # If multiple GPUs

    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def optuna_hp_space(trial):

    return {"scheduler": trial.suggest_categorical("scheduler",
        ["LinearLR", "ExponentialLR", "StepLR"]),
            "learning_rate": trial.suggest_float(1e-3, 1e-1, step=1e-2),
            "epochs": trail.suggest_categorical([8, 16, 32],
            "per_device_train_batch_size": trial.suggest_categorical([64, 128, 256]}



def compute_objective(trial):

    hp_space = optuna_hp_space(trial)

    set_seed(seed)
    training_args = define_training_args()

    hf_token = os.getenv("HF_TOKEN")
    privacy_engine = PrivacyEngine()#secure_mode=True)
    #------------------------------------------------------------------------
    dataset_dict = load_dataset("csv", data_dir=data_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dataset_dict = dataset_dict.map(
            lambda batch: tokenize_sentences(batch,
                tokenizer), batched=True)

    dataset_dict.set_format("torch", device=device)

    training_args = define_training_args()

    training_args.update(hp_space)

    train(**training_args)




def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    non_trainable_params = total_params - trainable_params

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params
    }


def train_one_epoch(epoch_index: int, tb_writer: SummaryWriter,
        model: BertForSequenceClassification, train_loader: DataLoader,
        optimizer, loss_fn=CrossEntropyLoss(reduction="none")) -> torch.Tensor:

    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, inputs in enumerate(train_loader):
        # Every data instance is an input + label pair

        inputs_ = inputs
        del inputs_["text"]

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(**inputs_)

        # Compute the loss and its gradients
        logits = outputs.logits

        labels = inputs_["labels"]
        del inputs_["labels"]

        loss = loss_fn(logits, labels)
        loss = loss.mean()

        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if (i + 1) % 10 == 0:
            last_loss = running_loss / 10  # loss per batch
            print('batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


def model_init(model_name: str, hf_token: str, num_labels: int) -> BertForSequenceClassification:

    model = BertForSequenceClassification.from_pretrained(
            model_name, token=hf_token, num_labels=num_labels) # , config=config)

    model.to(device)

    for param in model.base_model.parameters():
        param.requires_grad = False

    return model


def main(model_name: str, data_dir: str, seed: int, loss_fn=CrossEntropyLoss(reduction="none")):

    # Might need to test directly after training
    # TODO: Check when does it basically converge
    # TODO: Check bigger lr
    # TODO: Increase number of epochs 
    # TODO: Experiment with schedulers
    # TODO: Check num of predicted samples per class
    # TODO: Add a json file for all seeds
    

    study = optuna.create_study()
    study.optimize(compute_objective, n_trials=200)


def train(model: BertForSequenceClassification,
        learning_rate: float,
        num_train_epochs: int,
        per_device_train_batch_size: int,  
        train_dataset: Dataset,
        validation_dataset: Dataset,
        **kwargs):


    train_dataset = dataset_dict["train"]
    num_labels = len(train_dataset["labels"].unique())
    validation_dataset = dataset_dict["validation"]

    model = model_init(model_name, hf_token, num_labels)
    model_params = count_parameters(model)

    optimizer = get_optimizer(model, learning_rate)
    # scheduler = LinearLR(optimizer=optimizer) # , start_factor=0.1, gamma=0.01)
    train_loader = get_train_dataloader(train_dataset, train_batch_size)
    validation_loader = get_validation_loader(validation_dataset, validation_batch_size)

    #----------------------------------------------------------------------------

    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(training_args.logging_dir)

    EPOCHS = training_args.num_train_epochs

    model.train(True)

    model, optimizer, data_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.1,
    max_grad_norm=1.0,
    )

    #----------------------------------------------------------------------------

    best_vloss = 1_000_000

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data

        avg_loss = train_one_epoch(epoch, writer, model, train_loader, optimizer)
        # scheduler.step()

        epsilon = privacy_engine.get_epsilon(delta=10**-5)

        writer.add_scalars('Epsilon per Epoch', {"Epsilon": epsilon}, epoch + 1)

        writer.add_scalars('Random Seed', {"Seed": seed}, epoch + 1)

        best_model, epoch_best, best_vloss = evaluate_(model, epoch, validation_loader,
                writer, privacy_engine, best_vloss)

        if best_model is not None:

            model_params["best_model"] = best_model
            model_params["epoch_best"] = epoch_best
            model_params["best_vloss"] = best_vloss


        model_path = f"{training_args.output_dir}/seed={seed}"

        write_training_metadata(model_path, model_params)


def get_validation_dataloader(dataset: Dataset, batch_size: int) -> DataLoader:

    return DataLoader(validation_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            shuffle=True) # , collate_fn=data_collator)


def get_train_dataloader(dataset: Dataset, batch_size: int) -> DataLoader:

    return DataLoader(train_dataset,
            batch_size=training_args.per_device_train_batch_size,
            shuffle=True) # , collate_fn=data_collator)


def get_optimizer(model: BertForSequenceClassification, lr: float)
    return AdamW(model.classifier.parameters(), lr=lr, eps=1e-8)


def write_training_metadata(model_path: str, model_params: dict):

    os.makedirs(model_path, exist_ok=True)

    with open(
            os.path.join(
                model_path, "training_metadata.json"), "w") as f:

                json.dump(
                        {"opacus_epsilon": model_params["best_epsilon"],
                    "epoch": model_params["epoch_best"],
                    "trainable_params": model_params["trainable_params"], 
                    "total_params": model_params["total_params"],
                    "non_trainable_params": model_params["non_trainable_params"]
                    }, f)


    # best_model = convert_grad_sample_modules(best_model)

    best_model._module.save_pretrained(model_path, from_pt=True)
    model.train(True)
    print(f"(ε = {best_epsilon:.4f}, δ = {10**-5})")


def evaluate_(
        model: BertForSequenceClassification,
        epoch: int,
        validation_loader: DataLoader,
        writer: SummaryWriter,
        privacy_engine: PrivacyEngine,
        best_vloss: float) -> tuple[
        BertForSequenceClassification, int,
        float]:

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vinputs in enumerate(validation_loader):

            vinputs_ = vinputs

            del vinputs_["text"]

            voutputs = model(**vinputs_)
            vlogits = voutputs.logits

            vlabels = vinputs_["labels"]

            vloss = voutputs.loss

            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    {'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:

       best_epsilon = privacy_engine.get_epsilon(
               delta=10**-5)

       best_model = model
       epoch_best = epoch + 1
       best_vloss = avg_vloss

    else:
        best_model = None
        epoch_best = None
        best_vloss = None
        

   return best_model, epoch_best, best_vloss


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", help="The hf model name", type=str)

    parser.add_argument("--data_dir",
            help="The directory where the data reside", type=str)

    parser.add_argument("--seed", help="The random seed number to be used", type=int)

    args = vars(parser.parse_args())
    
    main(**args)
