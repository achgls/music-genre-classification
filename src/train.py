import os
from datetime import datetime
from typing import Union, Iterable

from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import GTZANDataset, ContaminatedGTZANDataset
from augmentations import WaveformAugment, SpecAugment
import utils


def train_one_epoch(
        model,
        transform,
        wav_aug,
        spec_aug,
        trn_loader,
        optimizer,
        loss_fn,
):
    running_loss = 0.
    pbar = tqdm(trn_loader)
    pbar.set_description_str("Training")

    correctly_classified: int = 0
    incorrectly_classified: int = 0

    for k, (x, y) in enumerate(pbar):
        optimizer.zero_grad()

        if wav_aug is not None: x = wav_aug(x)
        x = transform(x)
        if spec_aug is not None: x = spec_aug(x)

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

    avg_loss = float(running_loss / len(trn_loader))
    avg_acc = float(correctly_classified / (correctly_classified + incorrectly_classified))
    return avg_loss, avg_acc


def validate(
        model,
        transform,
        val_loader,
        loss_fn,
):
    running_loss = 0.
    pbar = tqdm(val_loader)
    pbar.set_description_str("Validation")

    correctly_classified: int = 0
    incorrectly_classified: int = 0

    with torch.no_grad():
        for k, (x, y) in enumerate(pbar, start=1):
            x = transform(x)

            logits = model(x)
            loss = loss_fn(logits, y)

            preds = torch.argmax(logits, dim=1)
            # print(torch.concatenate((preds.view(-1, 1), y.view(-1, 1)), dim=1))

            n_correct = torch.sum(preds == y)
            correctly_classified += n_correct
            incorrectly_classified += (preds.size(0) - n_correct)

            running_loss += loss
            avg_loss = running_loss / (k + 1)
            avg_acc = correctly_classified / (correctly_classified + incorrectly_classified)
            pbar.set_postfix_str(f"val. loss = {avg_loss:>6.4f} | val. accuracy = {avg_acc * 100:>5.2f} %")

    avg_loss = float(running_loss / len(val_loader))
    avg_acc = float(correctly_classified / (correctly_classified + incorrectly_classified))
    return avg_loss, avg_acc


def save_checkpoint(model, save_path):
    state_dict = model.state_dict()
    torch.save(state_dict, save_path)
    return state_dict


def train(
        num_epochs: int,
        model: nn.Module,
        transform: nn.Module,
        wav_aug: nn.Module,
        spec_aug: nn.Module,
        trn_loader: Union[torch.utils.data.DataLoader, Iterable],
        val_loader: Union[torch.utils.data.DataLoader, Iterable],
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        early_stopping: int,
        out_path: str,
        cp_freq: int,
):
    # Instead of summary writer, write to a CSV file
    best_val_loss = torch.inf
    best_val_accuracy = 0.

    output_tsv = os.path.join(out_path, "metrics.tsv")
    with open(output_tsv, 'w') as f:
        f.write('\t'.join(["epoch", "trn_loss", "val_loss", "trn_acc", "val_acc"])+'\n')

    epochs_without_improvement = 0

    for epoch in range(1, num_epochs+1):
        len_str = len("EPOCH") + len(str(epoch)) + 3
        len_cli = int(os.get_terminal_size().columns)
        size_bars = (len_cli - len_str) - 8
        print(u'\u2500' * 8 + " EPOCH %d " % epoch + u'\u2500' * size_bars)

        model.train()
        trn_loss, trn_accuracy = train_one_epoch(
            model=model,
            transform=transform,
            wav_aug=wav_aug,
            spec_aug=spec_aug,
            trn_loader=trn_loader,
            optimizer=optimizer,
            loss_fn=loss_fn)
        if scheduler is not None:
            scheduler.step()

        model.eval()
        val_loss, val_accuracy = validate(
            model=model,
            transform=transform,
            wav_aug=wav_aug,
            spec_aug=spec_aug,
            val_loader=val_loader,
            loss_fn=loss_fn
        )

        with open(output_tsv, 'a') as f:
            f.write('\t'.join([
                str(epoch),
                str(trn_loss),
                str(val_loss),
                str(trn_accuracy),
                str(val_accuracy)])+'\n')

        if val_loss < best_val_loss:
            save_checkpoint(model, os.path.join(out_path, "checkpoints", "best_loss.pt"))
            epochs_without_improvement = 0
            best_val_loss = val_loss
        elif val_accuracy > best_val_accuracy:
            save_checkpoint(model, os.path.join(out_path, "checkpoints", "best_acc_%d.pt" % round(val_accuracy)))
            epochs_without_improvement = 0
            best_val_accuracy = val_accuracy
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping:
                save_checkpoint(model, os.path.join(out_path, "checkpoints", "epoch%d_early_stopping.pt" % epoch))
                return model

        if epoch % cp_freq == 0:
            save_checkpoint(model, os.path.join(out_path, "checkpoints", "epoch%d.pt" % epoch))


def main(ns_args):
    # if config file is given, overwrites all other arguments
    if (config := ns_args.config_file) is not None:
        print(f"Using config from {config}, overwriting all other arguments.")
        with open(config, 'r') as f:
            config_args = json.load(f)

        ns_args = Namespace(**config_args)

    assert ns_args.num_fold is not None
    assert ns_args.model is not None
    assert ns_args.num_epochs is not None

    assert (batch_size := ns_args.batch_size) > 0 and isinstance(batch_size, int), \
        "Batch size must be a positive integer"

    cp_freq = ns_args.cp_freq
    if cp_freq is not None:
        assert cp_freq > 0 and isinstance(cp_freq, int), "Checkpoint frequency must be a positive integer."

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
    if (seed := ns_args.seed) is not None:
        torch.use_deterministic_algorithms(True)
        if device == torch.device("cuda"):
            torch.backends.cudnn.benchmark = False
        # Seeded model initialization
        print(f"Using seed {seed} for model initialization and data sampling.")
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

    wav_aug = None
    if (wav_aug_kwargs := ns_args.wav_aug) is not None:
        print(wav_aug_kwargs)
        wav_aug_kwargs = utils.parse_kwargs_arguments(wav_aug_kwargs)
        print(wav_aug_kwargs, type(wav_aug_kwargs))
        print("Apply data augmentation of waveform with following parameters:", wav_aug_kwargs)
        wav_aug = WaveformAugment(**wav_aug_kwargs)
    spec_aug = None
    if (spec_aug_kwargs := ns_args.spec_aug) is not None:
        print(spec_aug_kwargs)
        spec_aug_kwargs = utils.parse_kwargs_arguments(spec_aug_kwargs)
        print(wav_aug_kwargs, type(wav_aug_kwargs))
        print("Apply data augmentation of waveform with following parameters:", wav_aug_kwargs)
        spec_aug = SpecAugment(**spec_aug_kwargs)

    loss_kwargs = utils.parse_kwargs_arguments(ns_args.loss_kwargs)
    loss_fn = utils.get_loss(loss_name=ns_args.loss, **loss_kwargs)
    loss_fn = loss_fn.to(device)

    optimizer_kwargs = utils.parse_kwargs_arguments(ns_args.optimizer_kwargs)
    optimizer = utils.get_optimizer(optim_name=ns_args.optimizer, model=model, lr=ns_args.lr, **optimizer_kwargs)

    if (scheduler := ns_args.scheduler) is not None:
        scheduler_kwargs = utils.parse_kwargs_arguments(ns_args.scheduler_kwargs)
        scheduler = utils.get_scheduler(ns_args.scheduler, optimizer=optimizer, **scheduler_kwargs)

    # ----- Initialize data -----
    num_fold = ns_args.num_fold
    if ns_args.contaminated:
        data_class = ContaminatedGTZANDataset
        print("Using contaminated datasets.")
    else:
        data_class = GTZANDataset
        print("Using uncontaminated datasets.")

    trn_data = data_class(
        audio_dir=data_dir,
        num_fold=num_fold,
        overlap=0.5,
        sample_rate=22_050,
        win_duration=win_duration,
        file_duration=30.0,
        part="training",
        device=device)
    print(f"Using {len(np.unique(trn_data.index_files))} files for training, representing a total of {len(trn_data.start_offsets):,d} "
          f"{win_duration}-sec extracts.")

    val_data = data_class(
        audio_dir=data_dir,
        num_fold=num_fold,
        overlap=0.5,
        sample_rate=22_050,
        win_duration=win_duration,
        file_duration=30.0,
        part="validation",
        device=device)
    print(f"Using {len(np.unique(val_data.index_files))} files for validation, representing a total of {len(val_data.start_offsets):,d} "
          f"{win_duration}-sec extracts.")

    trn_loader = DataLoader(trn_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # ----- Initialize writing directory -----
    out_path = ns_args.out_path
    try:
        os.mkdir(out_path)
    except FileExistsError:
        pass

    out_path = os.path.join(out_path, ns_args.model)
    try:
        os.mkdir(out_path)
    except FileExistsError:
        pass

    run_tag = ns_args.run_tag
    run_id_list = [run_tag] if run_tag is not None else []
    run_id_list.extend(["fold%d" % num_fold, timestamp])
    run_id = "_".join(run_id_list)

    out_path = os.path.join(out_path, run_id)
    os.mkdir(out_path)
    os.mkdir(os.path.join(out_path, "checkpoints"))

    # Save experiments config as .json file
    with open(os.path.join(out_path, "config.json"), 'w') as f:
        json.dump(vars(ns_args), f, indent=2)
    print(f"Saved experiment config under {os.path.join(out_path, 'config.json')}")

    train(
        num_epochs=num_epochs,
        model=model,
        transform=transform,
        wav_aug=wav_aug,
        spec_aug=spec_aug,
        trn_loader=trn_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        early_stopping=early_stopping,
        out_path=out_path,
        cp_freq=cp_freq,
    )

    # Plot and save training and validation curve


if __name__ == "__main__":
    parser = ArgumentParser()

    print("Parsing arguments...", end=' ')
    parser.add_argument("--config-file", type=str, default=None,
                        help="If a config file is specified, all other parameters will be overwritten by the arguments "
                             "specified in the config file. If the original experiment was seeded, this should obtain "
                             "exactly the same results.")
    parser.add_argument("--data-dir", type=str, default="res/audio_data/")
    parser.add_argument("--contaminated", default=False, action="store_true",
                        help=f"If flag is present, will use 'contaminated' datasets with extracts from the same song "
                             f"being both in training and validation sets. Default: False, use uncontaminated data.")
    parser.add_argument("--slice-length", type=float, default=3.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-fold", type=int, default=None,
                        help="Index of fold to use as part of K-Fold cross-validation. From 1 to 5.")

    parser.add_argument("--out-path", type=str, default="results", help="Root path to store results.")
    parser.add_argument("--run-tag", type=str, default=None, help="Additional tag to label the current experiment.")
    parser.add_argument("--cp-freq", type=int, default=5, help="Number of epochs between model checkpoints.")
    parser.add_argument("--model-path", type=str, help="Model checkpoint path in case of warm start", default=None)

    parser.add_argument("-n", "--num-epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for parameter initialization and data sampling."
                             "Similar seed should lead to perfectly reproducible results under same parameters.")
    parser.add_argument("--early-stopping", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument("--model", type=str, help="type of model to use. required.", default=None)
    parser.add_argument("--model-kwargs", type=str, default=None)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default="Adam",
                        help="Type of optimizer to use (i.e. 'SGD' or 'Adam'). Default: Adam")
    parser.add_argument("--optimizer-kwargs", type=str, default=None)

    parser.add_argument("--loss", type=str, default="CrossEntropyLoss",
                        help="Type of loss to use. Default: Cross-Entropy")
    parser.add_argument("--loss-kwargs", type=str, default=None)

    parser.add_argument("--scheduler", type=str, default=None,
                        help="Type of scheduler to use for the learning rate decay.")
    parser.add_argument("--scheduler-kwargs", type=str, default=None)

    parser.add_argument("--feature", type=str, default="powerspec")
    parser.add_argument("--feature-kwargs", type=str, default=None)
    parser.add_argument("--spec-aug", type=str, default=None,
                        help="SpecAugment kwargs to feed the spectrogram augmentation module")
    parser.add_argument("--wav-aug", type=str, default=None,
                        help="kwargs to feed the waveform augmentation module, i.e min/max SNR and gain values")

    args = parser.parse_args()
    print("Done")

    main(args)
