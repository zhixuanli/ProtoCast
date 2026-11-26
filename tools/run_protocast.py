"""Train and evaluate ProtoCast on GlobalForecast and TyphoonTracking."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from protocast.config import BENCHMARK_DATA, learning_rate_tag, load_benchmark_config, validate_config

def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--benchmark', required=True, choices=tuple(BENCHMARK_DATA),
                   help='Benchmark name used in the paper')
    p.add_argument('--action', choices=('train', 'test'), default='train')
    p.add_argument('--data-root', required=True, type=Path)
    p.add_argument('--run-tag', default=os.environ.get('RUN_TAG'))
    p.add_argument('--output', type=Path)
    p.add_argument('--checkpoint', type=Path,
                   help='Evaluate a downloaded checkpoint with test')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--seed', type=int)
    p.add_argument('--lr', type=float)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--eval-batch-size', type=int)
    p.add_argument('--num-workers', type=int)
    p.add_argument('--precision', choices=('32-true', 'bf16-mixed'))
    p.add_argument('--wandb-mode', choices=('online', 'offline', 'disabled'),
                   default=os.environ.get('WANDB_MODE', 'disabled'),
                   help='Enable optional Weights & Biases logging (default: disabled)')
    p.add_argument('--dry-run', action='store_true')
    return p

def resolve(args):
    config = load_benchmark_config(ROOT / 'configs', args.benchmark)
    if args.seed is not None:
        config['seed'] = args.seed
    for name in ('lr', 'batch_size', 'eval_batch_size', 'num_workers', 'precision'):
        value = getattr(args, name)
        if value is not None:
            config['training'][name] = value
    method_id = 'ProtoCast'
    if args.lr is not None:
        method_id += '_lr' + learning_rate_tag(args.lr)
    config['experiment_id'] = method_id
    validate_config(config)
    if args.resume and args.action != 'train':
        raise ValueError('--resume is only for training')
    if args.checkpoint is not None and args.action != 'test':
        raise ValueError('--checkpoint is only for testing')
    run_tag = args.run_tag
    if run_tag is None:
        run_tag = 'evaluation' if args.checkpoint is not None else 'default'
    if not run_tag or run_tag in ('.', '..') or Path(run_tag).name != run_tag:
        raise ValueError('--run-tag must be one directory name')
    output = args.output or Path('work_dirs') / config['benchmark'] / method_id / run_tag
    output = ((ROOT / output) if not output.is_absolute() else output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Output must stay inside ProtoCast')
    return config, output

def main():
    args = parser().parse_args()
    config, output = resolve(args)
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint is not None else None
    print(json.dumps(dict(config=config, action=args.action, data_root=str(args.data_root.resolve()),
                         output=str(output), device=args.device, resume=args.resume, wandb_mode=args.wandb_mode,
                         checkpoint=str(checkpoint) if checkpoint is not None else None),
                     indent=2), flush=True)
    if args.dry_run:
        return
    from protocast.api.experiment import run_experiment
    run_experiment(config, args.data_root, output, args.device, args.action, args.resume, args.wandb_mode,
                   checkpoint=checkpoint)

if __name__ == '__main__':
    main()
