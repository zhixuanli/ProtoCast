"""Train or evaluate a paper baseline using its dataset-specific configuration."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = Path(__file__).resolve().parent / 'configs'


def load_data(config, data_root):
    from openstl.datasets.base_data import BaseDataModule

    options = {name: config[name] for name in (
        'batch_size', 'val_batch_size', 'num_workers', 'prefetch_factor',
        'use_augment', 'drop_last',
    )}
    if config['dataname'] == 'era5wind':
        from openstl.datasets.dataloader_era5_128_128 import load_data as load
        options.update(resolution=config['era5_resolution'], in_shape=config['in_shape'],
                       pre_seq_length=config['pre_seq_length'], aft_seq_length=config['aft_seq_length'])
    else:
        from openstl.datasets.dataloader_weather_npy import load_data as load
    return BaseDataModule(*load(data_root=data_root, **options))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train', 'test'])
    parser.add_argument('dataset', choices=sorted(p.name for p in CONFIGS.iterdir() if p.is_dir()))
    models = sorted({path.stem for path in CONFIGS.glob('*/*.json')})
    parser.add_argument('model', choices=models)
    parser.add_argument('--data-root', type=Path, help='Directory with the prepared dataset splits')
    parser.add_argument('--output-dir', type=Path, help='Training run directory containing checkpoints/')
    parser.add_argument('--gpu', type=int, default=0, help='Visible CUDA device index')
    args = parser.parse_args()

    config = json.loads((CONFIGS / args.dataset / (args.model + '.json')).read_text())
    data_root = args.data_root or ROOT / 'data' / args.dataset
    output = (args.output_dir or ROOT / 'work_dirs' / args.dataset / args.model).resolve()
    best = output / 'checkpoints' / 'best.ckpt'
    if args.action == 'test' and not best.is_file():
        parser.error(f'Train {args.model} on {args.dataset} first or select its run with --output-dir. Missing {best}')
    if args.action == 'train' and best.exists():
        parser.error(f'This run already has {best}. Select a new --output-dir for a new run.')

    import torch
    from lightning import Trainer, seed_everything
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from openstl.methods import method_maps

    seed_everything(config['seed'])
    data = load_data(config, data_root)
    model = method_maps[config['method']](
        **config, device=f'cuda:{args.gpu}', data_name=data.data_name,
        test_mean=data.test_mean, test_std=data.test_std,
        steps_per_epoch=len(data.train_loader) // config['accumulate_grad_batches'],
        save_dir=str(output),
    )
    checkpoint = ModelCheckpoint(dirpath=best.parent, filename='best', monitor='val_loss',
                                 mode='min', save_top_k=1, enable_version_counter=False)
    trainer = Trainer(
        accelerator='gpu', devices=[args.gpu], precision=config['precision'],
        max_epochs=config['epoch'], accumulate_grad_batches=config['accumulate_grad_batches'],
        default_root_dir=str(output), logger=CSVLogger(str(output), name='logs'),
        callbacks=[checkpoint],
    )
    if args.action == 'train':
        trainer.fit(model, datamodule=data)
    state = torch.load(best, map_location='cpu', weights_only=False)
    model.load_state_dict(state['state_dict'])
    trainer.test(model, datamodule=data)


if __name__ == '__main__':
    main()
