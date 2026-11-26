"""The fixed data and training contract shared by the ProtoCast CLI and loaders."""
import json
import math
from pathlib import Path


def load_benchmark_config(directory, benchmark):
    """Read the configuration named for the paper benchmark."""
    if benchmark not in BENCHMARK_DATA:
        raise ValueError(f'Unknown ProtoCast benchmark: {benchmark}')
    path = Path(directory) / f'{benchmark}.json'
    config = json.loads(path.read_text(encoding='utf-8'))
    if config['benchmark'] != benchmark:
        raise ValueError(f'{path.name} must declare benchmark {benchmark}')
    return config


def learning_rate_tag(value):
    """Canonical scientific notation for distinct learning-rate run directories."""
    mantissa, exponent = format(value, '.15e').split('e')
    return mantissa.rstrip('0').rstrip('.') + 'e' + str(int(exponent))

BENCHMARK_DATA = {
    'GlobalForecast': {
        'variables': ['u10', 'v10', 't2m', 'tcc', 'r'],
        'height': 32, 'width': 64, 'normalization': 'train_stats.json',
        'splits': {'train': [2010, 2015], 'val': [2016, 2016], 'test': [2017, 2018]},
        'metric_groups': {'uv10': ['u10', 'v10']},
    },
    'TyphoonTracking': {
        'variables': ['msl', 'u10', 'v10', 't2m'],
        'height': 32, 'width': 32, 'normalization': 'training_npz_minmax',
        'splits': {'train': [2017, 2021], 'val': [2022, 2022], 'test': [2023, 2023]},
        'metric_groups': {},
    },
}


VARIABLE_GROUPS = {'GlobalForecast': {'uv10': ['u10', 'v10'], 't2m': ['t2m'], 'tcc': ['tcc'], 'r': ['r']}, 'TyphoonTracking': {'msl': ['msl'], 'u10': ['u10'], 'v10': ['v10'], 't2m': ['t2m']}}

def validate_config(config, benchmark=None):
    """Validate the full ProtoCast data and training protocol."""
    def require(section, expected, prefix=''):
        for key, value in expected.items():
            if section.get(key) != value:
                raise ValueError(f'ProtoCast requires {prefix}{key}={value!r}; got {section.get(key)!r}')
    require(config, dict(method='ProtoCast', variant='entmax'))
    if benchmark is not None:
        require(config, {'benchmark': benchmark})
    if config['benchmark'] not in BENCHMARK_DATA:
        raise ValueError('Unknown ProtoCast benchmark')
    data = BENCHMARK_DATA[config['benchmark']]
    require(config['data'], dict(data, targets=data['variables'], input_steps=12,
                                output_steps=12, step_hours=1), 'data.')
    require(config['training'], dict(optimizer='adam', weight_decay=0, scheduler='cosine',
             warmup_epochs=0, accumulate_grad_batches=1, early_stopping=False,
             gradient_clip=None, loss='normalized_mse', latitude_weighting=False), 'training.')
    require(config['prototype'], {'refit_every': 1}, 'prototype.')
    expected = VARIABLE_GROUPS[config['benchmark']]
    if list(config['prototype']['variable_groups'].items()) != list(expected.items()):
        raise ValueError('ProtoCast variable_groups must preserve the benchmark channel order')
    if config['model'].get('post_adapter_fusion_gate_init') != 0.0:
        raise ValueError('ProtoCast requires a zero-initialized post-adapter gate')
    settings = config['training']
    if not math.isfinite(settings['lr']) or settings['lr'] <= 0:
        raise ValueError('training.lr must be finite and positive')
    if not math.isfinite(settings['min_lr']) or not 0 <= settings['min_lr'] <= settings['lr']:
        raise ValueError('training.min_lr must be between zero and training.lr')
    if config.get('evaluation_policy') is not None:
        raise ValueError('ProtoCast requires full best-checkpoint test after training')
    for key in ('epochs', 'batch_size', 'eval_batch_size', 'prefetch_factor'):
        if settings[key] < 1:
            raise ValueError(f'training.{key} must be positive')
    if settings['num_workers'] < 0:
        raise ValueError('num_workers must be nonnegative')
    if settings['precision'] not in ('32-true', 'bf16-mixed'):
        raise ValueError('precision must be 32-true or bf16-mixed')


def normalize_saved_config(config):
    """Identify a checkpoint that predates benchmark names by its data protocol."""
    if 'benchmark' in config:
        return config
    data = config['data']
    for benchmark, expected in BENCHMARK_DATA.items():
        if all(data.get(key) == value for key, value in expected.items()):
            normalized = dict(config, benchmark=benchmark)
            normalized.pop('table', None)
            return normalized
    raise ValueError('Saved configuration does not match a known benchmark data protocol')


def scientific_settings(config):
    """Names do not change the data, architecture or optimizer identity."""
    return {key: value for key, value in normalize_saved_config(config).items()
            if key not in ('method', 'version', 'source_version', 'experiment_id')}
