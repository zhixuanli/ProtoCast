import json
import numpy as np
import torch.nn as nn
import os.path as osp
import lightning as l
from openstl.core import get_optim_scheduler, timm_schedulers
from openstl.core.era5_metrics import ERA5Metrics, normalized_metrics_from_sums
from pathlib import Path

class Base_method(l.LightningModule):

    def __init__(self, **args):
        super().__init__()

        if args['dataname'] == 'era5wind':
            self.metric_list, self.spatial_norm = ['mse', 'mae', 'rmse', 'r2'], True
            self.channel_names = args['data_name']
        elif 'weather' in args['dataname']:
            self.metric_list, self.spatial_norm = args['metrics'], True
            self.channel_names = args['data_name'] if 'mv' in args['dataname'] else None
        else:
            self.metric_list, self.spatial_norm, self.channel_names = args['metrics'], False, None

        self.save_hyperparameters()
        self.model = self._build_model(**args)
        self.criterion = nn.MSELoss()
        self.test_outputs = []
        self._validation_metrics_active = False
        self.validation_results = {}

    def _build_model(self):
        raise NotImplementedError
    
    def configure_optimizers(self):
        optimizer, scheduler, by_epoch = get_optim_scheduler(
            self.hparams, 
            self.hparams.epoch, 
            self.model, 
            self.hparams.steps_per_epoch
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "epoch" if by_epoch else "step"
            },
        }
    
    def lr_scheduler_step(self, scheduler, metric):
        if any(isinstance(scheduler, sch) for sch in timm_schedulers):
            scheduler.step(epoch=self.current_epoch)
        else:
            if metric is None:
                scheduler.step()
            else:
                scheduler.step(metric)

    def forward(self, batch):
        NotImplementedError
    
    def training_step(self, batch, batch_idx):
        NotImplementedError

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x, batch_y)
        loss = self.criterion(pred_y, batch_y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        self._record_validation_batch(batch_x, pred_y, batch_y)
        return loss
    
    
    def on_validation_epoch_start(self):
        self.validation_results = {}
        self._validation_metrics_active = not self.trainer.sanity_checking
        if self._validation_metrics_active:
            self._start_metric_epoch('val')

    def _record_validation_batch(self, batch_x, pred_y, batch_y):
        if self._validation_metrics_active:
            self._record_metric_batch('val', pred_y, batch_y)

    def on_validation_epoch_end(self):
        if not self._validation_metrics_active:
            return
        self.validation_results = self._compute_metric_results('val')
        self.log_dict({'val/' + key: value for key, value in self.validation_results.items()},
                      on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self._validation_metrics_active = False

    def on_test_epoch_start(self):
        self.test_outputs.clear()
        self.test_results = {}
        self.test_normalized_results = {}
        self._start_metric_epoch('test')
        if self.hparams.dataname == 'era5wind':
            # Preserve the existing test accumulator interface.
            self.era5_metrics = self.test_era5_metrics

    def _start_metric_epoch(self, stage):
        # Distributed samplers can repeat windows; evaluate once on one device.
        if self.trainer.world_size != 1:
            raise ValueError('Evaluation metrics require single-device evaluation (--gpus 0)')
        if self.hparams.dataname == 'era5wind':
            setattr(self, stage + '_era5_metrics', ERA5Metrics(
                self.channel_names, self.hparams.test_mean, self.hparams.test_std))
        else:
            channels = len(self.channel_names) if self.channel_names is not None else 1
            setattr(self, stage + '_squared_error', np.zeros(channels, dtype=np.float64))
            setattr(self, stage + '_absolute_error', np.zeros(channels, dtype=np.float64))
            setattr(self, stage + '_count', 0)
            if 'r2' in self.metric_list:
                setattr(self, stage + '_target_mean', np.zeros(channels, dtype=np.float64))
                setattr(self, stage + '_target_m2', np.zeros(channels, dtype=np.float64))
                setattr(self, stage + '_target_count', 0)

    def _record_test_batch(self, batch_x, pred_y, batch_y):
        return self._record_metric_batch('test', pred_y, batch_y)

    def _record_metric_batch(self, stage, pred_y, batch_y):
        if self.hparams.dataname == 'era5wind':
            getattr(self, stage + '_era5_metrics').update(
                pred_y.detach().float().cpu().numpy(), batch_y.detach().float().cpu().numpy())
            return None
        pred = pred_y.detach().float().cpu().numpy().astype(np.float64)
        true = batch_y.detach().float().cpu().numpy().astype(np.float64)
        # Both current loaders normalize validation and test with training stats.
        if self.hparams.test_mean is not None and self.hparams.test_std is not None:
            pred = pred * self.hparams.test_std + self.hparams.test_mean
            true = true * self.hparams.test_std + self.hparams.test_mean

        # Retain existing channel selection and the non-weather spatial-sum scale.
        squared_error = getattr(self, stage + '_squared_error')
        absolute_error = getattr(self, stage + '_absolute_error')
        channels = len(squared_error)
        error = pred[:, :, :channels] - true[:, :, :channels]
        squared_error += np.sum(error ** 2, axis=(0, 1, 3, 4))
        absolute_error += np.sum(np.abs(error), axis=(0, 1, 3, 4))
        if 'r2' in self.metric_list:
            target = true[:, :, :channels]
            axes = (0, 1, 3, 4)
            target_count = target.shape[0] * target.shape[1] * target.shape[3] * target.shape[4]
            target_mean = target.mean(axis=axes)
            target_m2 = np.sum((target - target_mean[None, None, :, None, None]) ** 2, axis=axes)
            running_mean = getattr(self, stage + '_target_mean')
            running_m2 = getattr(self, stage + '_target_m2')
            running_count = getattr(self, stage + '_target_count')
            delta = target_mean - running_mean
            total = running_count + target_count
            running_m2 += target_m2 + delta ** 2 * running_count * target_count / total
            running_mean += delta * target_count / total
            setattr(self, stage + '_target_count', total)
        count = pred.shape[0] * pred.shape[1]
        if self.spatial_norm:
            count *= pred.shape[3] * pred.shape[4]
        setattr(self, stage + '_count', getattr(self, stage + '_count') + count)
        return None

    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x, batch_y)
        return self._record_test_batch(batch_x, pred_y, batch_y)

    def _compute_metric_results(self, stage, normalized=False):
        if self.hparams.dataname == 'era5wind':
            return getattr(self, stage + '_era5_metrics').compute(normalized=normalized)
        count = getattr(self, stage + '_count')
        if count == 0:
            raise ValueError(f'No {stage} samples were evaluated')
        squared_error = getattr(self, stage + '_squared_error')
        absolute_error = getattr(self, stage + '_absolute_error')
        target_m2 = getattr(self, stage + '_target_m2', None)
        results = {}
        names = self.channel_names if self.channel_names is not None else ['']
        if normalized:
            scale = (self.hparams.test_std
                     if self.hparams.test_mean is not None and self.hparams.test_std is not None
                     else np.ones(len(names)))
            return normalized_metrics_from_sums(
                squared_error, absolute_error, target_m2, count, scale, names,
                self.hparams.get('metric_groups', {}), rmse_prefix='rmse')
        for i, name in enumerate(names):
            mse = squared_error[i] / count
            results['mse_' + name] = float(mse)
            results['mae_' + name] = float(absolute_error[i] / count)
            results['rmse_' + name] = float(np.sqrt(mse))
            if 'r2' in self.metric_list:
                sse, sst = squared_error[i], target_m2[i]
                results['r2_' + name] = float(1.0 - sse / sst if sst > 0 else sse == 0)
        for name, members in self.hparams.get('metric_groups', {}).items():
            indices = [names.index(member) for member in members]
            group_count = count * len(indices)
            mse = squared_error[indices].sum() / group_count
            results['mse_' + name] = float(mse)
            results['mae_' + name] = float(absolute_error[indices].sum() / group_count)
            results['rmse_' + name] = float(np.sqrt(mse))
            if 'r2' in self.metric_list:
                # Each component uses its own mean: variance-weighted multi-output R2.
                sse = squared_error[indices].sum()
                sst = target_m2[indices].sum()
                results['r2_' + name] = float(1.0 - sse / sst if sst > 0 else sse == 0)
        return results

    def on_test_epoch_end(self):
        results = self._compute_metric_results('test')
        if self.hparams.dataname == 'era5wind':
            self.era5_test_results = results
            filename = 'era5_metrics.json'
        else:
            filename = 'metrics.json'

        self.test_results = results
        if self.hparams.dataname == 'era5wind' or 'weather' in self.hparams.dataname:
            self.test_normalized_results = self._compute_metric_results('test', normalized=True)
        if self.trainer.is_global_zero:
            print(', '.join(f'{name}:{value}' for name, value in results.items()))
            folder_path = Path(self.hparams.save_dir) / 'saved'
            folder_path.mkdir(parents=True, exist_ok=True)
            with open(osp.join(folder_path, filename), 'w') as handle:
                json.dump(results, handle, indent=2, allow_nan=False)
            if self.test_normalized_results:
                normalized_filename = filename.replace('.json', '_normalized.json')
                with open(osp.join(folder_path, normalized_filename), 'w') as handle:
                    json.dump(self.test_normalized_results, handle, indent=2, allow_nan=False)
        return results
