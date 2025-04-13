"""
Finetune the models with diffential privacy contraints
The models to be checked are BERT, RoBERTa, BioBERT, ClinicalBERT, MentalBERT, MentalRoBERTa
"""

import argparse
from validate_model_performance import main
from transformers import Trainer, TrainingArguments, AutoModel, DataCollatorWithPadding
from datasets import Dataset
from typing import Callable
from opacus.privacy_engine import PrivacyEngine
import torch


class DPTrainer(Trainer):


    def __init__(self, model: AutoModel,
            args: TrainingArguments,
            data_collator: DataCollatorWithPadding,
            train_dataset: Dataset,
            eval_dataset: Dataset,
            compute_metrics: Callable[tuple[torch.Tensor, torch.Tensor], dict]): 
        
        super().__init__(model=model, args=args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset= eval_dataset,
                compute_metrics=compute_metrics)

        optimizer = torch.optim.AdamW(
                self.model.classifier.parameters())

        train_dataloader = super().get_train_dataloader()
        eval_dataloader = super().get_eval_dataloader()

        self.create_optimizer()
        self.model.train()

        self.privacy_engine = PrivacyEngine()

        model, optimizer, train_dataloader = self.make_private(train_dataloader)

        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader


    def get_train_dataloader(self):

        return self.train_dataloader


    def get_eval_dataloader(self):

        return self.eval_dataloader


    def make_private(self, data_loader):

        breakpoint()
        model, optimizer, data_loader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=data_loader,
            noise_multiplier=1.1,
            max_grad_norm=1.0,
        )

        return model, optimizer, data_loader



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", help="The hf model name", type=str)

    parser.add_argument("--data_dir",
            help="The directory where the data reside", type=str)

    args = parser.parse_args()
    
    model_name = args.model_name
    data_dir = args.data_dir

    main(model_name= model_name, data_dir= data_dir, trainer_cls=DPTrainer)
