import os
from datetime import datetime
from typing import Union, Iterable

import torchaudio.transforms
from tqdm import tqdm
from argparse import ArgumentParser
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import GTZANDataset
import utils


def train_one_epoch(
        model,
        transform,
        trn_loader,
        optimizer,
        loss_fn,
):
    running_loss = 0.
    pbar = tqdm(trn_loader)

    correctly_classified: int = 0
    incorrectly_classified: int = 0

    for k, (x, y) in enumerate(pbar):
        optimizer.zero_grad()

        x = transform(x)
        logits = model(x)
        loss = loss_fn(logits, y)

        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        # print(torch.concatenate((preds.view(-1, 1), y.view(-1, 1)), dim=1))

        n_correct = torch.sum(preds == y)
        correctly_classified += n_correct
        incorrectly_classified += (preds.size(0) - n_correct)

        running_loss += loss.item()
        avg_loss = running_loss / (k + 1)
        avg_acc = correctly_classified / (correctly_classified + incorrectly_classified)
        pbar.set_postfix_str(f"loss = {avg_loss:>6.4f} | accuracy = {avg_acc * 100:>5.2f} %")

    avg_loss = running_loss / len(trn_loader)
    avg_acc = correctly_classified / (correctly_classified + incorrectly_classified)
    return avg_loss, avg_acc


def validate(
        model,
        transform,
        val_loader,
        loss_fn,
):
    running_loss = 0.
    pbar = tqdm(val_loader)

    correctly_classified: int = 0
    incorrectly_classified: int = 0

    with torch.no_grad():
        for k, (x, y) in enumerate(pbar, start=1):
            x = transform(x)
            logits = model(x)
            loss = loss_fn(logits, y)

            preds = torch.argmax(logits, dim=1)
            # # print(torch.concatenate((preds.view(-1, 1), y.view(-1, 1)), dim=1))

            n_correct = torch.sum(preds == y)
            correctly_classified += n_correct
            incorrectly_classified += (preds.size(0) - n_correct)

            running_loss += loss
            avg_loss = running_loss / (k + 1)
            avg_acc = correctly_classified / (correctly_classified + incorrectly_classified)
            pbar.set_postfix_str(f"val. loss = {avg_loss:>6.4f} | val. accuracy = {avg_acc * 100:>5.2f} %")

    avg_loss = running_loss / len(val_loader)
    avg_acc = correctly_classified / (correctly_classified + incorrectly_classified)
    return avg_loss, avg_acc


def save_checkpoint(model, save_path):
    state_dict = model.state_dict()
    torch.save(state_dict, save_path)
    return state_dict


def train(
        num_epochs: int,
        model: nn.Module,
        transform: nn.Module,
        trn_loader: Union[torch.utils.data.DataLoader, Iterable],
        val_loader: Union[torch.utils.data.DataLoader, Iterable],
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        early_stopping: int,
        out_path: str
):
    # Instead of summary writer, write to a CSV file
    best_val_loss = torch.inf
    best_val_accuracy = 0.

    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        trn_loss, trn_accuracy = train_one_epoch(
            model=model,
            transform=transform,
            trn_loader=trn_loader,
            optimizer=optimizer,
            loss_fn=loss_fn)
        scheduler.step()
        print(trn_loss, trn_accuracy)

        model.eval()
        val_loss, val_accuracy = validate(
            model=model,
            transform=transform,
            val_loader=val_loader,
            loss_fn=loss_fn
        )
        print(val_loss, val_accuracy)

        # if val_loss < best_val_loss:
        #     save_checkpoint(model, ...)
        #     epochs_without_improvement = 0
        # elif val_accuracy < best_val_accuracy:
        #     save_checkpoint(model, ...)
        #     epochs_without_improvement = 0
        # else:
        #     epochs_without_improvement += 1
        #     if epochs_without_improvement >= early_stopping:
        #         save_checkpoint(model, ...)
        #         return model


def main(ns_args):
    assert (batch_size := ns_args.batch_size) > 0, "Batch size must be a positive integer"

    assert os.path.isdir(data_dir := ns_args.data_dir), "Unrecognized data directory"
    assert 0.0 < (win_duration := ns_args.slice_length) <= 30.0,\
        "Frame duration should be positive and smaller than length of the whole song extract (30sec)"

    assert 0 < (num_epochs := ns_args.num_epochs) <= 1000
    if (early_stopping := ns_args.early_stopping) is not None:
        assert 0 <= early_stopping < num_epochs, "Number of epochs for early stopping must be in range [0; num_epochs["

    # ----- Initialize timestamp for logging -----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ----- Initialize device -----
    device = ns_args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # ----- Set manual seed for complete reproducibility -----
    if seed := ns_args.seed is not None:
        # Seeded model initialization
        assert isinstance(seed, int)
        torch.manual_seed(seed)

    # ----- Initialize model -----
    model_kwargs = utils.parse_kwargs_arguments(ns_args.model_kwargs)
    model = utils.get_model(model_name=ns_args.model, num_classes=10, **model_kwargs)
    model = model.to(device)
    # model.compile()  # Use PyTorch Lighting to compile model

    # ----- Initialize loss, optimizer and scheduler -----
    feature_kwargs = utils.parse_kwargs_arguments(ns_args.feature_kwargs)
    transform = utils.get_transform(feature_name=ns_args.feature, **feature_kwargs)
    transform = transform.to(device)

    loss_kwargs = utils.parse_kwargs_arguments(ns_args.loss_kwargs)
    loss_fn = utils.get_loss(loss_name=ns_args.loss, **loss_kwargs)
    loss_fn = loss_fn.to(device)

    optimizer_kwargs = utils.parse_kwargs_arguments(ns_args.optimizer_kwargs)
    optimizer = utils.get_optimizer(optim_name=ns_args.optimizer, model=model, lr=ns_args.lr, **optimizer_kwargs)

    if scheduler := ns_args.scheduler is not None:
        scheduler_kwargs = utils.parse_kwargs_arguments(ns_args.scheduler_kwargs)
        scheduler = utils.get_scheduler(ns_args.scheduler, optimizer=optimizer, **scheduler_kwargs)

    # ----- Initialize data -----
    trn_data = GTZANDataset(
        audio_dir=data_dir,
        num_fold=ns_args.num_fold,
        overlap=0.5,
        sample_rate=22_050,
        win_duration=win_duration,
        file_duration=30.0,
        part="training",
        device=device
    )
    val_data = GTZANDataset(
        audio_dir=data_dir,
        num_fold=ns_args.num_fold,
        overlap=0.5,
        sample_rate=22_050,
        win_duration=win_duration,
        file_duration=30.0,
        part="validation",
        device=device
    )

    trn_loader = DataLoader(trn_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    train(
        num_epochs=num_epochs,
        model=model,
        transform=transform,
        trn_loader=trn_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        early_stopping=early_stopping,
        out_path='',
    )


if __name__ == "__main__":
    parser = ArgumentParser()

    print("Parsing arguments...", end=' ')
    parser.add_argument("--data-dir", type=str, default="res/audio_data/")
    parser.add_argument("--slice-length", type=float, default=3.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-fold", type=int, required=True,
                        help="Index of fold to use as part of K-Fold cross-validation. From 1 to 5.")

    parser.add_argument("--out-path", type=str, default=None)
    parser.add_argument("--model-path", type=str, help="Model checkpoint path in case of warm start", default=None)

    parser.add_argument("-n", "--num-epochs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for parameter initialization and data sampling."
                             "Similar seed should lead to perfectly reproducible results under same parameters.")
    parser.add_argument("--early-stopping", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument("--model", type=str, help="type of model to use", required=True)
    parser.add_argument("--model-kwargs", type=str, default=None)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default="Adam",
                        help="Type of optimizer to use (i.e. 'SGD' or 'Adam'). Default: Adam")
    parser.add_argument("--optimizer-kwargs", type=str, default=None)

    parser.add_argument("--loss", type=str, default="CrossEntropyLoss",
                        help="Type of loss to use. Default: Cross-Entropy")
    parser.add_argument("--loss-kwargs", type=str, default=None)

    parser.add_argument("--scheduler", type=str, default="LinearLR",
                        help="Type of scheduler to use for the learning rate decay. Default: linear decay.")
    parser.add_argument("--scheduler-kwargs", type=str, default=None)

    parser.add_argument("--feature", type=str, default="spec")
    parser.add_argument("--feature-kwargs", type=str, default=None)
    parser.add_argument("--data-aug", action="store_true")

    args = parser.parse_args()
    print("Done")

    main(args)
