from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from openstl.datasets.utils import create_loader


class ERA5Dataset(Dataset):
    """Released raw CTHW ERA5 arrays, normalized with training-file statistics."""

    def __init__(self, npz_path, seq_len=12, pred_len=12, step_size=1, stats=None):
        if min(seq_len, pred_len, step_size) < 1:
            raise ValueError('Sequence lengths and step_size must be positive')
        with np.load(npz_path, allow_pickle=True) as loaded:
            self.data = loaded['data']
            self.stats = loaded['stats'].item() if stats is None else stats
        if self.data.ndim != 4 or self.data.shape[0] != 4:
            raise ValueError('ERA5 data must have shape (4, time, height, width)')

        self.data_name = ['msl', 'u10', 'v10', 't2m']
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.step_size = step_size
        self.total_timesteps = max(
            0, (self.data.shape[1] - seq_len - pred_len) // step_size + 1)
        self.min_values = np.asarray(
            [self.stats[var]['min'] for var in self.data_name],
            dtype=np.float32).reshape(1, 4, 1, 1)
        self.max_values = np.asarray(
            [self.stats[var]['max'] for var in self.data_name],
            dtype=np.float32).reshape(1, 4, 1, 1)
        # BaseDataModule uses these names for affine inverse offset/scale.
        # For ERA5 they are min and range, not a mean and standard deviation.
        self.mean = self.min_values[None]
        self.std = (self.max_values - self.min_values + 1e-8)[None]

    def __len__(self):
        return self.total_timesteps

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        start = idx * self.step_size
        stop = start + self.seq_len + self.pred_len
        # Normalize only the requested window instead of a second full data copy.
        sequence = self.data[:, start:stop].transpose(1, 0, 2, 3)
        sequence = (sequence - self.mean[0]) / self.std[0]
        return (torch.as_tensor(sequence[:self.seq_len], dtype=torch.float32),
                torch.as_tensor(sequence[self.seq_len:], dtype=torch.float32))

    def inverse_transform(self, values):
        """Restore physical units for a TCHW or BTCHW array/tensor."""
        if values.ndim not in (4, 5) or values.shape[-3] != 4:
            raise ValueError('Expected TCHW or BTCHW values with four channels')
        offset = self.mean[0] if values.ndim == 4 else self.mean
        scale = self.std[0] if values.ndim == 4 else self.std
        if torch.is_tensor(values):
            offset = torch.as_tensor(offset, dtype=values.dtype, device=values.device)
            scale = torch.as_tensor(scale, dtype=values.dtype, device=values.device)
        return values * scale + offset


def load_data(batch_size, val_batch_size, data_root, num_workers=4,
              pre_seq_length=12, aft_seq_length=12, in_shape=None,
              distributed=False, use_augment=False, use_prefetcher=False,
              drop_last=False, train_stride=1, val_stride=1, test_stride=1,
              resolution=128, prefetch_factor=2):
    """data_root contains train/val/test_{resolution}_{resolution}.npz.

    Stride 1 yields the paper's 43801/8737 train/test windows for 12 -> 12.
    This is the revised reproduction setting, not evidence of the author's run.
    """
    if resolution not in (32, 128):
        raise ValueError('ERA5 resolution must be 32 or 128')
    root = Path(data_root)
    suffix = f'{resolution}_{resolution}.npz'
    train_set = ERA5Dataset(root / f'train_{suffix}', pre_seq_length,
                            aft_seq_length, train_stride)
    val_set = ERA5Dataset(root / f'val_{suffix}', pre_seq_length,
                          aft_seq_length, val_stride, stats=train_set.stats)
    test_set = ERA5Dataset(root / f'test_{suffix}', pre_seq_length,
                           aft_seq_length, test_stride, stats=train_set.stats)

    loaders = []
    for dataset, size, training in ((train_set, batch_size, True),
                                     (val_set, val_batch_size, False),
                                     (test_set, val_batch_size, False)):
        if in_shape is not None:
            actual_shape = (dataset.data.shape[0], *dataset.data.shape[-2:])
            if actual_shape != tuple(in_shape[1:]):
                raise ValueError(f'ERA5 data shape {actual_shape} does not match in_shape {in_shape}')
        loaders.append(create_loader(
            dataset, batch_size=size, shuffle=training, is_training=training,
            num_workers=num_workers, distributed=distributed,
            use_prefetcher=use_prefetcher, input_channels=4, pin_memory=True,
            persistent_workers=num_workers > 0, prefetch_factor=prefetch_factor,
            drop_last=training))
    return tuple(loaders)
