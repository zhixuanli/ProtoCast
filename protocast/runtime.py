"""Project-local ProtoCast output and cache paths, initialized before heavyweight imports."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def output_directory(path, root=ROOT):
    path = Path(path).expanduser()
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(root):
        raise ValueError('ProtoCast outputs must stay inside ProtoCast')
    return path


def runtime(output, root=ROOT):
    output = output_directory(output, root)
    output.mkdir(parents=True, exist_ok=True)
    for variable, path in {
        'XDG_CACHE_HOME': root / '.cache', 'TORCH_HOME': root / '.cache/torch',
        'TMPDIR': root / '.runtime/tmp',
        'WANDB_DIR': output, 'WANDB_CACHE_DIR': root / '.cache/wandb',
        'WANDB_CONFIG_DIR': root / '.runtime/wandb-config',
        'WANDB_DATA_DIR': root / '.runtime/wandb-data',
    }.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    return output
