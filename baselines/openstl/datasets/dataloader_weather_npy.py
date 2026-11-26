"""GlobalForecast: raw TCHW arrays and train-only JSON statistics."""
import json
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import Dataset
from .utils import create_loader

CHANNELS = ['u10', 'v10', 't2m', 'tcc', 'r']


class WeatherBenchNPYDataset(Dataset):
    """Read consecutive observation and forecast windows from raw NPY arrays."""

    def __init__(self, path, mean, std, idx_in, idx_out, use_augment=False):
        self.data = np.load(path, mmap_mode='r', allow_pickle=False)
        if self.data.ndim != 4 or self.data.shape[1:] != (5, 32, 64):
            raise ValueError(f'{path}: expected (T, 5, 32, 64), got {self.data.shape}')
        if self.data.dtype != np.float32:
            raise ValueError(f'{path}: expected raw float32 data, got {self.data.dtype}')
        self.data_name = CHANNELS.copy()
        self.mean, self.std = mean, std
        self.idx_in, self.idx_out = np.asarray(idx_in), np.asarray(idx_out)
        if use_augment:
            raise ValueError("The baseline protocol uses unaugmented weather windows.")
        # NPY already omits the final source hour. Preserve the NetCDF loader's
        # window endpoints (T-24 for 12->12), without cropping another frame.
        self.valid_idx = np.arange(-self.idx_in[0], len(self.data) - self.idx_out[-1] - 1)
        if not len(self.valid_idx):
            raise ValueError(f'{path}: not enough frames for the requested windows')

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, index):
        center = self.valid_idx[index]
        data = torch.from_numpy(((self.data[center + self.idx_in] - self.mean) / self.std).astype(np.float32))
        labels = torch.from_numpy(((self.data[center + self.idx_out] - self.mean) / self.std).astype(np.float32))
        return data, labels


def load_data(batch_size, val_batch_size, data_root, num_workers=4,
              idx_in=tuple(range(-11, 1)), idx_out=tuple(range(1, 13)),
              distributed=False, use_augment=False, use_prefetcher=False,
              drop_last=False, prefetch_factor=2, **kwargs):
    root = Path(data_root)
    with (root / 'train_stats.json').open(encoding='utf-8') as handle:
        stats = json.load(handle)
    values = np.asarray([stats[name] for name in CHANNELS], dtype=np.float64)
    if values.shape != (5, 2) or not np.isfinite(values).all() or (values[:, 1] <= 0).any():
        raise ValueError('train_stats.json must contain finite [mean, positive std] for each channel')
    mean = values[:, 0].reshape(1, 5, 1, 1)
    std = values[:, 1].reshape(1, 5, 1, 1)
    loaders = []
    for split in ('train', 'val', 'test'):
        training = split == 'train'
        dataset = WeatherBenchNPYDataset(
            root / f'{split}.npy', mean, std, idx_in, idx_out,
            use_augment=use_augment if training else False)
        loaders.append(create_loader(
            dataset, batch_size=batch_size if training else val_batch_size,
            shuffle=training, is_training=training, pin_memory=True,
            drop_last=True if training else drop_last, num_workers=num_workers,
            persistent_workers=num_workers > 0, distributed=distributed,
            prefetch_factor=prefetch_factor,
            use_prefetcher=use_prefetcher))
    return tuple(loaders)
