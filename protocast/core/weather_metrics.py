"""Global weather metrics with the current GlobalForecast/TyphoonTracking JSON key contract."""
import numpy as np


class WeatherMetrics:
    """Accumulate channel sums and centered target variance, never predictions."""

    def __init__(self, meta):
        self.channel_names = list(meta['channel_names'])
        self.metric_groups = dict(meta.get('metric_groups', {}))
        self.rmse_prefix = 'rmse' if meta['benchmark'] == 'GlobalForecast' else 'rmse_'
        channels = len(self.channel_names)
        self.offset = np.asarray(meta['offset'], dtype=np.float64).reshape(1, 1, channels, 1, 1)
        self.scale = np.asarray(meta['scale'], dtype=np.float64).reshape(1, 1, channels, 1, 1)
        if not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ValueError('Normalization scales must be finite and positive')
        self.count = 0
        self.squared_error = np.zeros(channels, dtype=np.float64)
        self.absolute_error = np.zeros(channels, dtype=np.float64)
        self.target_mean = np.zeros(channels, dtype=np.float64)
        self.target_m2 = np.zeros(channels, dtype=np.float64)

    def update(self, pred, true):
        """Add one batch of normalized NumPy BTCHW predictions and targets."""
        pred = np.asarray(pred, dtype=np.float64)
        true = np.asarray(true, dtype=np.float64)
        if pred.shape != true.shape or pred.ndim != 5 or pred.shape[2] != len(self.channel_names):
            raise ValueError('Expected matching BTCHW arrays with one channel per weather variable')
        count = pred.shape[0] * pred.shape[1] * pred.shape[3] * pred.shape[4]
        if count == 0:
            return
        pred = pred * self.scale + self.offset
        true = true * self.scale + self.offset
        axes = (0, 1, 3, 4)
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
        """Return physical metrics, or metrics in the loader's normalized space."""
        if self.count == 0:
            raise ValueError('No weather samples were evaluated')
        scale = self.scale.reshape(-1) if normalized else np.ones(len(self.channel_names))
        squared = self.squared_error / scale ** 2
        absolute = self.absolute_error / scale
        variance = self.target_m2 / scale ** 2
        result = {}

        def add_metrics(name, indices, rmse_key):
            denominator = self.count * len(indices)
            sse = squared[indices].sum()
            sst = variance[indices].sum()
            mse = sse / denominator
            result['mse_' + name] = float(mse)
            result['mae_' + name] = float(absolute[indices].sum() / denominator)
            result[rmse_key] = float(np.sqrt(mse))
            # Match the current constant-target convention: perfect=1, else=0.
            result['r2_' + name] = float(1.0 - sse / sst if sst > 0 else sse == 0)

        for index, name in enumerate(self.channel_names):
            add_metrics(name, [index], self.rmse_prefix + name)
        for name, members in self.metric_groups.items():
            indices = [self.channel_names.index(member) for member in members]
            add_metrics(name, indices, 'rmse_' + name)
        return result
