"""Train and evaluate ProtoCast on GlobalForecast and TyphoonTracking."""
import contextlib
import json
import os
import random
import time
from pathlib import Path
from uuid import uuid4
import numpy as np
import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
from protocast.config import scientific_settings
from protocast.core.weather_metrics import WeatherMetrics
from protocast.core.weather_loss import forecast_mse, forecast_squared_error
from protocast.datasets.weather import load_benchmark_data
from protocast.utils.prototype_sampling import refit_plan
from protocast.runtime import runtime
from protocast.utils.checkpoint import dump_json, load_checkpoint, save_checkpoint
ROOT = Path(__file__).resolve().parents[2]

def validate_config(config):
    from protocast.config import validate_config as validate
    validate(config)

def build_model(config):
    from protocast.models import build_protocast
    return build_protocast(config)

def build_optimizer(model, config):
    settings = config['training']
    options = dict(lr=settings['lr'], weight_decay=0.0)
    if 'adam_beta1' in settings or 'adam_beta2' in settings:
        options['betas'] = (settings.get('adam_beta1', 0.9), settings.get('adam_beta2', 0.999))
    return torch.optim.Adam(model.parameters(), **options)

@torch.no_grad()
def refit_bank(model, dataset, plan, config, device):
    was_training = model.training
    model.eval()
    features = {name: [] for name in model.variable_groups}
    try:
        batch_size = config['training']['batch_size']
        for start in range(0, len(plan), batch_size):
            chunk = plan[start:start + batch_size]
            x = torch.stack([dataset[w][0] for w, _, _ in chunk]).to(device)
            _, descriptions = model.prefix(x)
            for i, (_, patches, leads) in enumerate(chunk):
                patches = torch.as_tensor(patches, device=device)
                for name, description in descriptions.items():
                    selected = description[i, patches]
                    features[name].append(selected.float().cpu())
        sampled = {name: torch.cat(values)[:config['prototype']['init_samples']] for name, values in features.items()}
        return model.refit_banks(sampled, config['prototype']['kmeans_iters'], config['seed'])
    finally:
        model.train(was_training)

def autocast(device, precision):
    if precision == 'bf16-mixed':
        if device.type != 'cuda' or not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 requires a supported CUDA device; use 32-true for CPU checks')
        return torch.autocast('cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()

def _require_finite_gradients(parameters):
    """Reject nonfinite gradient elements without clipping or an overflow-prone L2 sum."""
    return torch.nn.utils.clip_grad_norm_(parameters, float('inf'), norm_type=float('inf'), error_if_nonfinite=True)

def make_loaders(data_root, config, splits):
    train = config['training']
    return load_benchmark_data(data_root, config['benchmark'], batch_size=train['batch_size'], eval_batch_size=train['eval_batch_size'], num_workers=train['num_workers'], prefetch_factor=train['prefetch_factor'], seed=config['seed'], splits=splits)

@torch.no_grad()
def evaluate(model, loader, meta, device, precision):
    model.eval()
    accumulator = WeatherMetrics(meta)
    squared, elements = (0.0, 0)
    for x, y in loader:
        x, y = (x.to(device, non_blocking=True), y.to(device, non_blocking=True))
        with autocast(device, precision):
            prediction = model(x)
        errors = forecast_squared_error(prediction, y)
        squared += float(errors.sum(dtype=torch.float64))
        elements += errors.numel()
        accumulator.update(prediction.float().cpu().numpy(), y.float().cpu().numpy())
    if not elements:
        raise ValueError('Evaluation split has no windows')
    values = dict(loss=squared / elements, physical=accumulator.compute(), normalized=accumulator.compute(normalized=True))
    if not np.isfinite([values['loss'], *values['physical'].values(), *values['normalized'].values()]).all():
        raise FloatingPointError('Nonfinite evaluation result')
    return values

def start_wandb(config, output, mode, resume=False, testing=False):
    if mode == 'disabled':
        return None
    import wandb
    project = os.environ.get('WANDB_PROJECT', 'ProtoCast')
    entity = os.environ.get('WANDB_ENTITY') or None
    metadata = output / ('wandb-test-run.json' if testing else 'wandb-run.json')
    if resume and metadata.exists():
        saved = json.loads(metadata.read_text())
        if saved['project'] != project or saved['entity'] != entity:
            raise ValueError('Resume with the original W&B project/entity')
        run_id = saved['id']
    else:
        run_id = uuid4().hex
        dump_json(metadata, dict(id=run_id, project=project, entity=entity))
    run = wandb.init(project=project, entity=entity, id=run_id, mode=mode, resume=('allow' if resume else 'never') if mode == 'online' else None, dir=str(output), name=f"{config.get('experiment_id', 'ProtoCast')}/{config['benchmark']}/{output.name}" + ('/test' if testing else ''), group=f"{config['method']}/{config['benchmark']}", job_type='test' if testing else 'train', config=config, save_code=False, settings=wandb.Settings(disable_git=True, disable_code=True, save_code=False, console='off'))
    dump_json(metadata, dict(id=run_id, project=project, entity=entity, url=run.url))
    return run

def checkpoint_payload(model, optimizer, scheduler, epoch, best, step, config, meta, loader):
    return dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), epoch=epoch, best=best, global_step=step, config=config, normalization=meta, rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, loader_rng=loader.generator.get_state())

def _evaluation_settings(config):
    settings = scientific_settings(config)
    # These options affect how evaluation runs, independently of the saved model.
    settings['training'] = {key: value for key, value in settings['training'].items()
                            if key not in ('eval_batch_size', 'num_workers', 'prefetch_factor', 'precision')}
    return settings

def test_best(model, config, data_root, output, device, precision, run=None, *, checkpoint=None):
    checkpoint_path = Path(checkpoint) if checkpoint is not None else output / 'checkpoints/best.ckpt'
    best = load_checkpoint(checkpoint_path, device)
    compare = _evaluation_settings if checkpoint is not None else scientific_settings
    if compare(best['config']) != compare(config):
        raise ValueError('Checkpoint and requested benchmark or experiment configuration differ')
    model.load_state_dict(best['model'], strict=True)
    loaders, meta = make_loaders(data_root, config, ('test',))
    verify_normalization(best['normalization'], meta)
    result = evaluate(model, loaders['test'], meta, device, precision)
    saved = output / 'saved'
    saved.mkdir(exist_ok=True)
    dump_json(saved / 'metrics.json', result['physical'])
    dump_json(saved / 'metrics_normalized.json', result['normalized'])
    dump_json(saved / 'normalization.json', meta)
    checkpoint_record = str(checkpoint_path) if checkpoint is not None else 'checkpoints/best.ckpt'
    dump_json(saved / 'test_record.json', dict(method='ProtoCast', experiment_id=config.get('experiment_id', 'ProtoCast'), initial_lr=config['training']['lr'], benchmark=config['benchmark'], best_epoch=best['epoch'] + 1, checkpoint=checkpoint_record, normalized_mse=result['loss'], test_windows=len(loaders['test'].dataset)))
    if run is not None:
        run.log({**{'test/' + k: v for k, v in result['physical'].items()}, **{'test_normalized/' + k: v for k, v in result['normalized'].items()}})
    return result

def require_completed_training(config, output):
    """The best epoch may be early, but the full training budget must be finished."""
    last = load_checkpoint(output / 'checkpoints/last.ckpt', 'cpu')
    if scientific_settings(last['config']) != scientific_settings(config):
        raise ValueError('Completed checkpoint and requested configuration differ')
    if last['epoch'] + 1 != config['training']['epochs']:
        raise ValueError('Formal test is locked until all configured training epochs complete')

def verify_normalization(expected, actual):
    for key in ('benchmark', 'channel_names', 'offset', 'scale', 'metric_groups', 'normalization'):
        if expected[key] != actual[key]:
            raise ValueError(f'Dataset normalization changed: {key}')
    if expected.get('latitude_weighting') != actual.get('latitude_weighting'):
        raise ValueError('Dataset normalization changed: latitude_weighting')

def run_experiment(config, data_root, output, device='cuda:0', action='train', resume=False, wandb_mode='disabled', *,
                   checkpoint=None):
    """Train a run or evaluate its validation-selected checkpoint."""
    if action not in ('train', 'test'):
        raise ValueError(f'Unknown ProtoCast action: {action}')
    if checkpoint is not None:
        if action != 'test':
            raise ValueError('A checkpoint path is only supported for testing')
        checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    validate_config(config)
    output = runtime(output, ROOT)
    device = torch.device(device)
    settings = config['training']
    precision = settings['precision']
    with autocast(device, precision):
        pass
    if checkpoint is None:
        lock = output / 'experiment.json'
        if lock.exists() and scientific_settings(json.loads(lock.read_text())) != scientific_settings(config):
            raise ValueError('Use a new run directory when experiment settings change')
        dump_json(lock, config)
    model = build_model(config).to(device)
    random.seed(config['seed'])
    if action == 'test':
        if checkpoint is None:
            require_completed_training(config, output)
        run = start_wandb(config, output, wandb_mode, testing=True)
        try:
            result = test_best(model, config, data_root, output, device, precision, run, checkpoint=checkpoint)
            if checkpoint is None:
                dump_json(output / 'progress.json', dict(method='ProtoCast', benchmark=config['benchmark'], status='complete', epoch=settings['epochs'], epochs=settings['epochs']))
            return result
        except (Exception, KeyboardInterrupt):
            if run is not None:
                run.finish(exit_code=1)
                run = None
            raise
        finally:
            if run is not None:
                run.finish()
    last = output / 'checkpoints/last.ckpt'
    if last.exists() and (not resume):
        raise ValueError('This run already has a checkpoint; use --resume or a new run tag')
    if resume and (not last.exists()):
        raise FileNotFoundError(last)
    loaders, meta = make_loaders(data_root, config, ('train', 'val'))
    if not len(loaders['train']):
        raise ValueError('Training split is smaller than the training batch size')
    optimizer = build_optimizer(model, config)
    scheduler = CosineLRScheduler(optimizer, t_initial=settings['epochs'], lr_min=settings['min_lr'], warmup_t=0, t_in_epochs=True)
    start, best, step = (0, float('inf'), 0)
    if resume:
        saved = load_checkpoint(last, device)
        if scientific_settings(saved['config']) != scientific_settings(config):
            raise ValueError('Resume configuration differs from the checkpoint')
        verify_normalization(saved['normalization'], meta)
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        start, best, step = (saved['epoch'] + 1, saved['best'], saved['global_step'])
        if loaders['train'].persistent_workers:
            iter(loaders['train'])
        torch.set_rng_state(saved['rng'].cpu())
        np.random.set_state(saved['numpy_rng'])
        random.setstate(saved['python_rng'])
        loaders['train'].generator.set_state(saved['loader_rng'].cpu())
        if device.type == 'cuda' and saved['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng']])
    dump_json(output / 'data_record.json', meta)
    plan = refit_plan(model, loaders['train'].dataset, config)
    run = start_wandb(config, output, wandb_mode, resume=resume)
    checkpoints = output / 'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    try:
        for epoch in range(start, settings['epochs']):
            started = time.monotonic()
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            bank = refit_bank(model, loaders['train'].dataset, plan, config, device)
            with (output / 'bank_history.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(epoch=epoch + 1, **bank)) + '\n')
            model.train()
            total, elements = (0.0, 0)
            learning_rates = dict(lr=optimizer.param_groups[0]['lr'])
            for index, (x, y) in enumerate(loaders['train']):
                x, y = (x.to(device, non_blocking=True), y.to(device, non_blocking=True))
                optimizer.zero_grad(set_to_none=True)
                with autocast(device, precision):
                    prediction = model(x)
                    loss = forecast_mse(prediction, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at epoch {epoch + 1}, batch {index + 1}')
                loss.backward()
                _require_finite_gradients(model.parameters())
                optimizer.step()
                step += 1
                total += float(loss.detach()) * y.numel()
                elements += y.numel()
                if index == 0 or (index + 1) % 10 == 0:
                    progress = dict(method='ProtoCast', benchmark=config['benchmark'], status='running', phase='train', epoch=epoch + 1, epochs=settings['epochs'], batch=index + 1, batches=len(loaders['train']), global_step=step, train_loss=total / elements, **learning_rates, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None, peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None, elapsed_seconds=time.monotonic() - started)
                    dump_json(output / 'progress.json', progress)
                    print(json.dumps(progress), flush=True)
                    if run is not None:
                        run.log(dict(epoch=epoch + 1, global_step=step, train_loss=float(loss.detach()), **learning_rates))
            validation = evaluate(model, loaders['val'], meta, device, precision)
            improved = validation['loss'] < best
            if improved:
                best = validation['loss']
            scheduler.step(epoch=epoch)
            payload = checkpoint_payload(model, optimizer, scheduler, epoch, best, step, config, meta, loaders['train'])
            if improved:
                save_checkpoint(checkpoints / 'best.ckpt', payload)
            save_checkpoint(last, payload)
            record = dict(epoch=epoch + 1, train_loss=total / elements, val_loss=validation['loss'], best_val_loss=best, **learning_rates, global_step=step, seconds=time.monotonic() - started, validation_metrics=validation['physical'])
            record.update(peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None, peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None)
            with (output / 'history.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
            if run is not None:
                run.log({**{k: v for k, v in record.items() if k != 'validation_metrics'},
                         **{'val/' + k: v for k, v in validation['physical'].items()}})
        dump_json(output / 'progress.json', dict(status='running', phase='best_checkpoint_test', epoch=settings['epochs'], epochs=settings['epochs']))
        result = test_best(model, config, data_root, output, device, precision, run)
        dump_json(output / 'progress.json', dict(method='ProtoCast', benchmark=config['benchmark'], status='complete', epoch=settings['epochs'], epochs=settings['epochs']))
        return result
    except (Exception, KeyboardInterrupt) as error:
        status = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        dump_json(output / 'progress.json', dict(status=status, error=str(error) or type(error).__name__))
        if run is not None:
            run.finish(exit_code=1)
            run = None
        raise
    finally:
        if run is not None:
            run.finish()
