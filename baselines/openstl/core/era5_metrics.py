"""Pooled weather metrics in physical and normalized units."""
import numpy as np


def normalized_metrics_from_sums(squared_error, absolute_error, target_m2,
                                 count, scale, channel_names, metric_groups=None,
                                 rmse_prefix='rmse_'):
    """Convert physical sums to metrics in the loader's normalized space.

    ``count`` is the number of elements per channel. ``scale`` contains one
    normalization scale per channel (singleton axes are accepted). Group R2
    uses each component's own target mean and normalized variance weighting.
    The input sums are never modified, so physical metrics remain available.
    """
    names = list(channel_names)
    scales = np.broadcast_to(np.asarray(scale, dtype=np.float64).reshape(-1),
                             (len(names),))
    if count <= 0:
        raise ValueError('No samples were evaluated')
    if not np.all(np.isfinite(scales)) or np.any(scales == 0):
        raise ValueError('Normalization scales must be finite and nonzero')
    squared = np.asarray(squared_error, dtype=np.float64) / scales ** 2
    absolute = np.asarray(absolute_error, dtype=np.float64) / np.abs(scales)
    variance = None if target_m2 is None else np.asarray(target_m2, dtype=np.float64) / scales ** 2
    result = {}

    def add_metrics(name, indices, rmse_key):
        denominator = count * len(indices)
        sse = squared[indices].sum()
        mse = sse / denominator
        result['mse_' + name] = float(mse)
        result['mae_' + name] = float(absolute[indices].sum() / denominator)
        result[rmse_key] = float(np.sqrt(mse))
        if variance is not None:
            sst = variance[indices].sum()
            result['r2_' + name] = float(1.0 - sse / sst if sst > 0 else sse == 0)

    for i, name in enumerate(names):
        add_metrics(name, [i], rmse_prefix + name)
    for name, members in (metric_groups or {}).items():
        add_metrics(name, [names.index(member) for member in members], 'rmse_' + name)
    return result


class ERA5Metrics:
    def __init__(self, channel_names, offset, scale):
        self.channel_names = list(channel_names)
        self.offset = np.asarray(offset, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.count = 0
        self.squared_error = np.zeros(len(channel_names), dtype=np.float64)
        self.absolute_error = np.zeros(len(channel_names), dtype=np.float64)
        self.target_mean = np.zeros(len(channel_names), dtype=np.float64)
        self.target_m2 = np.zeros(len(channel_names), dtype=np.float64)

    def update(self, pred, true):
        pred = np.asarray(pred, dtype=np.float64) * self.scale + self.offset
        true = np.asarray(true, dtype=np.float64) * self.scale + self.offset
        if pred.shape != true.shape or pred.ndim != 5 or pred.shape[2] != len(self.channel_names):
            raise ValueError('Expected matching BTCHW arrays with one channel per ERA5 variable')
        axes = (0, 1, 3, 4)
        count = pred.shape[0] * pred.shape[1] * pred.shape[3] * pred.shape[4]
        error = pred - true
        self.squared_error += np.sum(error ** 2, axis=axes)
        self.absolute_error += np.sum(np.abs(error), axis=axes)
        mean = true.mean(axis=axes)
        m2 = np.sum((true - mean[None, None, :, None, None]) ** 2, axis=axes)
        delta = mean - self.target_mean
        total = self.count + count
        self.target_m2 += m2 + delta ** 2 * self.count * count / total
        self.target_mean += delta * count / total
        self.count = total

    def compute(self, normalized=False):
        if normalized:
            return normalized_metrics_from_sums(
                self.squared_error, self.absolute_error, self.target_m2,
                self.count, self.scale, self.channel_names)
        if self.count == 0:
            raise ValueError('No ERA5 test samples were evaluated')
        result = {}
        for i, name in enumerate(self.channel_names):
            mse = self.squared_error[i] / self.count
            result['mse_' + name] = float(mse)
            result['mae_' + name] = float(self.absolute_error[i] / self.count)
            result['rmse_' + name] = float(np.sqrt(mse))
            # Constant-target convention: perfect predictions score 1, others 0.
            r2 = (1.0 - self.squared_error[i] / self.target_m2[i]
                  if self.target_m2[i] > 0 else float(self.squared_error[i] == 0))
            result['r2_' + name] = float(r2)
        return result
