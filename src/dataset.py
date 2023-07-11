import os
from glob import glob
from typing import List, Tuple
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
from torchaudio import load, info
from torch.utils.data import Dataset


_genres_ = ('blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock')

_label_dict_ = {
    genre: k for k, genre in enumerate(_genres_)
}


def get_KFolds(data_dir: str, n_folds: int, seed: int = None, format='wav') -> List[Tuple[np.ndarray]]:
    filenames = np.array(sorted(glob(os.path.join(data_dir, '*', f'*.{format}'))))
    genres = np.array([os.path.split(os.path.split(fn)[0])[1] for fn in filenames])
    splits = [
        (filenames[split[0]], filenames[split[1]])
        for split in StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(filenames, genres)
    ]
    return splits


class GTZANDataset(Dataset):
    _config_ = {
        "split_seed": 123456789,
        "n_folds": 5,
    }

    def __init__(
            self,
            audio_dir: str,
            num_fold: int,
            overlap: float = 0.5,
            sample_rate=22_050,
            win_duration: float = 3.0,
            file_duration: float = None,
            part="training",
            device="auto"
    ):
        if isinstance(device, str):
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            raise AssertionError("Device must be a string or a torch.device object")

        assert 1 <= num_fold <= self._config_["n_folds"]
        filenames = get_KFolds(
            data_dir=audio_dir,
            n_folds=self._config_["n_folds"],
            seed=self._config_["split_seed"],
            format="wav"
        )[num_fold - 1][0 if part == "training" else 1]
        self.files = filenames

        self.sample_rate = sample_rate
        self.overlap = overlap
        self.window_duration = win_duration
        self.extract_length = round(self.window_duration * self.sample_rate)
        self.file_duration = file_duration

        index = self._compute_index()

        self.index_files = index[:, 0].astype(str)
        self.start_offsets = index[:, 1].astype(int)
        self.to_pad = index[:, 2].astype(int)
        self.labels = torch.tensor(index[:, 3].astype(int)).to(self.device, dtype=torch.int64)

        self.pad_fn = self.hold_padding

    @staticmethod
    def hold_padding(wav, pad_len):
        if pad_len < 0:
            wav = torch.nn.functional.pad(wav, (0, -pad_len), mode="replicate")
        return wav

    @staticmethod
    def _get_label_from_file(fn, indices=True):
        label = os.path.split(os.path.split(fn)[0])[1]
        if indices:
            label = _label_dict_[label]
        return label

    def _compute_index(self):
        index = []
        extract_length = round(self.window_duration * self.sample_rate)
        hop_length = round(extract_length * (1 - self.overlap))
        if self.file_duration is not None:
            max_start_offset = round(self.file_duration * self.sample_rate) - extract_length
        else:
            max_start_offset = np.inf

        for fn in self.files:
            label = self._get_label_from_file(fn, indices=True)
            file_length = info(fn).num_frames
            start_offset = 0
            delta = file_length
            while delta > 0 and start_offset <= max_start_offset:
                to_pad = 0
                delta = file_length - (start_offset + extract_length)
                if delta < 0:
                    to_pad = delta
                index.append([fn, start_offset, to_pad, label])
                start_offset += hop_length

        return np.array(index)

    def __len__(self):
        return len(self.index_files)

    def __getitem__(self, idx):
        wav = load(
            self.index_files[idx],
            frame_offset=self.start_offsets[idx],
            num_frames=self.extract_length)[0]
        wav = self.pad_fn(wav, self.to_pad[idx])
        return wav.to(self.device), self.labels[idx]


class ContaminatedGTZANDataset(GTZANDataset):
    _config_ = {
        "split_seed": 123456789,
        "n_folds": 3,
    }

    def __init__(
            self,
            audio_dir: str,
            num_fold: int,
            overlap: float = 0.5,
            sample_rate=22_050,
            win_duration: float = 3.0,
            file_duration: float = None,
            part="training",
            device="auto"
    ):
        if isinstance(device, str):
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            raise AssertionError("Device must be a string or a torch.device object")

        assert 1 <= num_fold <= self._config_["n_folds"]

        filenames = np.array(glob(os.path.join(audio_dir, '*', f'*.wav')))
        self.files = filenames

        self.sample_rate = sample_rate
        self.overlap = overlap
        self.window_duration = win_duration
        self.extract_length = round(self.window_duration * self.sample_rate)
        self.file_duration = file_duration

        index = self._compute_index()

        split_idx = [
            split for split in StratifiedKFold(n_splits=3, shuffle=True, random_state=12345).split(
                X=np.arange(index.shape[0]), y=index[:, 3].astype(int))
        ][num_fold - 1][0 if part == "training" else 1]

        self.index_files = index[split_idx, 0].astype(str)
        self.start_offsets = index[split_idx, 1].astype(int)
        self.to_pad = index[split_idx, 2].astype(int)
        self.labels = torch.tensor(index[split_idx, 3].astype(int)).to(self.device, dtype=torch.int64)

        self.pad_fn = self.hold_padding

