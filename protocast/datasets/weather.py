"""ProtoCast loaders for the current GlobalForecast and TyphoonTracking low-resolution data."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from protocast.config import BENCHMARK_DATA


BENCHMARKS = {benchmark: dict(channel_names=data['variables'],
                     shape=(len(data['variables']), data['height'], data['width']),
                     normalization='zscore' if benchmark == 'GlobalForecast' else 'minmax_0_1',
                     metric_groups=data['metric_groups'])
          for benchmark, data in BENCHMARK_DATA.items()}


class WeatherDataset(Dataset):
    """Raw data with 12 -> 12, stride-one windows and training normalization."""

    def __init__(self, path, benchmark, offset, scale):
        self.benchmark = benchmark
        self.data_name = BENCHMARKS[benchmark]['channel_names'].copy()
        dtype = np.float64 if benchmark == 'GlobalForecast' else np.float32
        self.offset = np.asarray(offset, dtype=dtype).reshape(1, -1, 1, 1)
        self.scale = np.asarray(scale, dtype=dtype).reshape(1, -1, 1, 1)
        if benchmark == 'GlobalForecast':
            self.data = np.load(path, mmap_mode='r', allow_pickle=False)
            if self.data.ndim != 4 or self.data.shape[1:] != BENCHMARKS[benchmark]['shape']:
                raise ValueError(f'{path}: expected (T, 5, 32, 64), got {self.data.shape}')
            if self.data.dtype != np.float32:
                raise ValueError(f'{path}: expected raw float32 data, got {self.data.dtype}')
            # The prepared NPY already omits the final source hour. Preserve the
            # established T-24 windows without cropping another source frame.
            self.window_count = self.data.shape[0] - 24
        else:
            with np.load(path, allow_pickle=True) as loaded:
                self.data = loaded['data']
            actual = (self.data.shape[0], *self.data.shape[2:]) if self.data.ndim == 4 else None
            if actual != BENCHMARKS[benchmark]['shape']:
                raise ValueError(f'{path}: expected (4, T, 32, 32), got {self.data.shape}')
            self.window_count = self.data.shape[1] - 23
        if self.window_count <= 0:
            raise ValueError(f'{path}: not enough frames for the requested windows')

    def __len__(self):
        return self.window_count

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if self.benchmark == 'GlobalForecast':
            sequence = self.data[index:index + 24]
        else:
            sequence = self.data[:, index:index + 24].transpose(1, 0, 2, 3)
        sequence = ((sequence - self.offset) / self.scale).astype(np.float32)
        return torch.from_numpy(sequence[:12]), torch.from_numpy(sequence[12:])


def load_benchmark_data(data_root, benchmark, batch_size=128, eval_batch_size=64,
                    num_workers=4, prefetch_factor=2, seed=42,
                    splits=('train', 'val')):
    """Return requested split loaders and JSON-compatible protocol metadata.

    GlobalForecast expects raw TCHW ``{split}.npy`` and ``train_stats.json``.
    TyphoonTracking expects raw CTHW ``{split}_32_32.npz``; only the training
    archive's statistics are used. No test data is opened unless requested.
    """
    if benchmark not in BENCHMARKS:
        raise ValueError(f'Unknown ProtoCast benchmark: {benchmark}')
    splits = tuple(splits)
    if not splits or any(split not in ('train', 'val', 'test') for split in splits):
        raise ValueError('splits must contain train, val, or test')
    root = Path(data_root)
    spec = BENCHMARKS[benchmark]
    names = spec['channel_names']
    if benchmark == 'GlobalForecast':
        with (root / 'train_stats.json').open(encoding='utf-8') as handle:
            stats = json.load(handle)
        values = np.asarray([stats[name] for name in names], dtype=np.float64)
        if values.shape != (len(names), 2):
            raise ValueError('train_stats.json must contain [mean, std] for each channel')
        offset, scale = values[:, 0], values[:, 1]
    else:
        # NPZ members are lazy: a test-only call does not decompress train data.
        with np.load(root / 'train_32_32.npz', allow_pickle=True) as loaded:
            stats = loaded['stats'].item()
        offset = np.asarray([stats[name]['min'] for name in names], dtype=np.float32)
        maximum = np.asarray([stats[name]['max'] for name in names], dtype=np.float32)
        scale = maximum - offset + 1e-8
    if not np.isfinite(offset).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Training normalization must have finite offsets and positive scales')

    meta = {
        'benchmark': benchmark,
        'channel_names': names.copy(),
        'offset': offset.tolist(),
        'scale': scale.tolist(),
        'normalization': spec['normalization'],
        'metric_groups': {name: members.copy() for name, members in spec['metric_groups'].items()},
        'window_counts': {},
    }
    loaders = {}
    for split in splits:
        training = split == 'train'
        filename = f'{split}.npy' if benchmark == 'GlobalForecast' else f'{split}_32_32.npz'
        dataset = WeatherDataset(root / filename, benchmark, offset, scale)
        options = dict(
            batch_size=batch_size if training else eval_batch_size,
            shuffle=training, drop_last=training, num_workers=num_workers,
            pin_memory=True, persistent_workers=num_workers > 0,
            generator=torch.Generator().manual_seed(seed),
        )
        if num_workers > 0:
            options['prefetch_factor'] = prefetch_factor
        loaders[split] = DataLoader(dataset, **options)
        meta['window_counts'][split] = len(dataset)
    return loaders, meta
