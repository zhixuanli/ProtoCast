<div align="center">
  <a href="https://huggingface.co/papers/2611.25848">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/654a5412ebec0c55a5a96253/UA3_Ij0bY-I4unMRj4Rch.gif" width="100%" alt="ProtoCast weather forecasting demo">
  </a>
</div>

# ProtoCast: Interpretable Prototype-Conditioned Adaptation for Weather Forecasting

> ProtoCast uses group-specific weather prototypes to route low-rank corrections in multivariate forecasts. It takes 12 hourly observations and predicts the next 12 hours for every input channel.

[![Paper on arXiv](https://img.shields.io/badge/Paper-arXiv-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2611.25848)
[![Project Page](https://img.shields.io/badge/Project%20Page-2457A6?style=for-the-badge)](https://zhixuanli.github.io/projects/li2026protocast/)
[![WeatherForecast Dataset](https://img.shields.io/badge/Dataset-WeatherForecast-0F766E?style=for-the-badge)](https://huggingface.co/datasets/OliverLee520/WeatherForecast)
<br>
[![Hugging Face Paper](https://img.shields.io/badge/Hugging%20Face%20Paper-FCD34D?style=for-the-badge&logo=huggingface&logoColor=20252B)](https://huggingface.co/papers/2611.25848)
[![Hugging Face Model Weights](https://img.shields.io/badge/Hugging%20Face-Model%20Weights-6D28D9?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/OliverLee520/ProtoCast)

## Media

Chinese Media Coverage

- [将门创投](https://mp.weixin.qq.com/s/94FFZo1WEWZ_fbha3vB-SA)
- [我爱计算机视觉](https://mp.weixin.qq.com/s/OZdYpXwR-J4a2o9O7djCuw)
- [极市平台](https://mp.weixin.qq.com/s/pQOFsUxLDO4gKypUeJXFdA)

## Contents

- [Requirements](#requirements)
- [Train and Evaluate](#train-and-evaluate)
- [Baselines](#baselines)
- [Paper Run Logs and Checkpoints](#paper-run-logs-and-checkpoints)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)
- [License](#license)

## Requirements

### Environment Setup

- Linux x86-64 or WSL2
- NVIDIA GPU with a driver that supports CUDA 12.8
- Python 3.10.8 in the `protocast` Conda environment
- PyTorch 2.7.0 (CUDA 12.8) and timm 0.9.16 in [requirements.txt](requirements.txt)

```bash
conda env create -f environment.yml
conda activate protocast
python -m pip install --no-deps .
python -c 'import torch, timm; print(torch.__version__, torch.version.cuda, timm.__version__); assert torch.cuda.is_available()'
```

<a id="dataset-preparetion"></a>

### Dataset Preparation

The paper uses the prepared weather data from [WeatherForecast on Hugging Face](https://huggingface.co/datasets/OliverLee520/WeatherForecast). The links and commands below use the same dataset revision.

| Benchmark | Local data root |
| --- | --- |
| GlobalForecast (32 × 64) | [`data/GlobalForecast`](https://huggingface.co/datasets/OliverLee520/WeatherForecast/tree/f999fd0f3e866d806c258b347e1efb99511eccdb/GlobalForecast) |
| TyphoonTracking (32 × 32) | [`data/TyphoonTracking`](https://huggingface.co/datasets/OliverLee520/WeatherForecast/tree/f999fd0f3e866d806c258b347e1efb99511eccdb/TyphoonTracking) |

From the repository root, use the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) to download the benchmark you want to run.

<details>
<summary>Dataset Download Commands</summary>

```bash
DATA_REPO=OliverLee520/WeatherForecast
DATA_REVISION=f999fd0f3e866d806c258b347e1efb99511eccdb

# GlobalForecast (32 × 64)
hf download "$DATA_REPO" --repo-type dataset --revision "$DATA_REVISION" --local-dir data \
  GlobalForecast/train.npy GlobalForecast/val.npy GlobalForecast/test.npy \
  GlobalForecast/train_stats.json

# TyphoonTracking (32 × 32)
hf download "$DATA_REPO" --repo-type dataset --revision "$DATA_REVISION" --local-dir data \
  TyphoonTracking/train_32_32.npz TyphoonTracking/val_32_32.npz TyphoonTracking/test_32_32.npz
```

</details>

## Train and Evaluate

### Train

```bash
# GlobalForecast
bash scripts/GlobalForecast/ProtoCast.sh train data/GlobalForecast

# TyphoonTracking
bash scripts/TyphoonTracking/ProtoCast.sh train data/TyphoonTracking
```

The run scripts use `python` from the active environment. Set `PYTHON_BIN` to select a different interpreter, or `GPU_ID` to select a different GPU.

To resume an interrupted run, reuse its data path and run tag.

```bash
# GlobalForecast
bash scripts/GlobalForecast/ProtoCast.sh train data/GlobalForecast --resume

# TyphoonTracking
bash scripts/TyphoonTracking/ProtoCast.sh train data/TyphoonTracking --resume
```

Outputs go to `work_dirs/<benchmark>/ProtoCast/default/` by default. Use `--run-tag` for a fresh output directory and reuse it on resume and test. The run records its settings in `experiment.json` and saves both checkpoints under `checkpoints/`. Use `--dry-run` to print the resolved settings.

Weights & Biases logging is opt-in. Use `--wandb-mode offline` or `--wandb-mode online` after installing the logging extra (`python -m pip install '.[logging]'`), or from the Conda environment above.

### Evaluate

Download the matching `protocast.ckpt` from the Weights column below. The examples place it in `weights/GlobalForecast/` or `weights/TyphoonTracking/` under the repository root. If you save it elsewhere, pass its absolute path to `--checkpoint`. Evaluation uses that single checkpoint and the dataset's test split and training normalization files.

- GlobalForecast uses `test.npy` and `train_stats.json`.
- TyphoonTracking uses `test_32_32.npz` and reads the normalization statistics from `train_32_32.npz`.

```bash
# GlobalForecast
bash scripts/GlobalForecast/ProtoCast.sh test data/GlobalForecast --checkpoint weights/GlobalForecast/protocast.ckpt

# TyphoonTracking
bash scripts/TyphoonTracking/ProtoCast.sh test data/TyphoonTracking --checkpoint weights/TyphoonTracking/protocast.ckpt
```

Results for a downloaded checkpoint go to `work_dirs/<benchmark>/ProtoCast/evaluation/` by default. Evaluation writes `saved/test_record.json`, `saved/metrics.json` (physical units), and `saved/metrics_normalized.json`. Metrics include MSE, MAE, RMSE, and R². For GlobalForecast, UV10 pools the two wind channels under the paper's aggregation protocol.

To reevaluate a completed local training run, omit `--checkpoint` and reuse the run tag from training.

## Baselines

The ten paper baselines use the same prepared datasets. From the repository root, activate the `protocast` environment and install the additional dependencies:

```bash
python -m pip install -r baselines/requirements.txt
# PredFormer only: fetch the upstream model implementation once
bash baselines/prepare_predformer.sh
```

Choose a model with `python baselines/run.py --help`. For example, train and evaluate Met²Net with:

```bash
# GlobalForecast
python baselines/run.py train GlobalForecast Met2Net
python baselines/run.py test GlobalForecast Met2Net

# TyphoonTracking
python baselines/run.py train TyphoonTracking Met2Net
python baselines/run.py test TyphoonTracking Met2Net
```

Training saves the checkpoint with the lowest validation loss to `work_dirs/<dataset>/<model>/checkpoints/best.ckpt`. The test command loads that local checkpoint and writes metrics to the run's `saved/` directory.

## Paper Run Logs and Checkpoints

These are the original ProtoCast runs used for the two main paper results.

| Dataset | LOG | Weights |
| --- | --- | --- |
| GlobalForecast | [LOG](https://forge.coreweave.com/wandb/zhixuanli/ProtoCast/runs/b3ca5b8cfe99404081bafead606cbedc) | [protocast.ckpt](https://huggingface.co/OliverLee520/ProtoCast/resolve/main/GlobalForecast/protocast.ckpt) |
| TyphoonTracking | [LOG](https://forge.coreweave.com/wandb/zhixuanli/ProtoCast/runs/663c6f5c9b2a4eaab383acd5cec1984a) | [protocast.ckpt](https://huggingface.co/OliverLee520/ProtoCast/resolve/main/TyphoonTracking/protocast.ckpt) |

### Local Directory Layout

Place the downloaded data in `data/` and each paper checkpoint in its matching dataset folder under `weights/`.

<details>
<summary>View the Directory Tree</summary>

```text
ProtoCast/
├── data/
│   ├── GlobalForecast/
│   │   ├── train.npy
│   │   ├── val.npy
│   │   ├── test.npy
│   │   └── train_stats.json
│   └── TyphoonTracking/
│       ├── train_32_32.npz
│       ├── val_32_32.npz
│       └── test_32_32.npz
├── weights/
│   ├── GlobalForecast/protocast.ckpt
│   └── TyphoonTracking/protocast.ckpt
├── baselines/
├── configs/
├── protocast/
├── scripts/
├── tools/
├── .gitignore
├── CITATION.cff
├── environment.yml
├── LICENSE
├── MANIFEST.in
├── pyproject.toml
├── README.md
├── requirements.txt
└── THIRD_PARTY_NOTICES.md
```

</details>

## Citation

If you use ProtoCast in your research, please cite our paper.

```bibtex
@article{li2026protocast,
  title={Interpretable Prototype-Conditioned Adaptation for Weather Forecasting},
  author={Li, Zhixuan and Jiang, Hao and Hui, Chen and Geng, Kai and Lin, Weisi},
  journal={arXiv preprint arXiv:2611.25848},
  year={2026}
}
```

For the software citation, see [CITATION.cff](CITATION.cff).

## Acknowledgement

We thank authors of [PredFormer](https://github.com/yyyujintang/PredFormer), [Met2Net](https://github.com/ShremG/Met2Net), and [OpenSTL](https://github.com/chengtan9907/OpenSTL) for sharing their code with the community.

## License

See [LICENSE](LICENSE) for Apache-2.0 and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for code and dataset attribution. The [dataset terms](https://huggingface.co/datasets/OliverLee520/WeatherForecast/blob/f999fd0f3e866d806c258b347e1efb99511eccdb/LICENSE.md) apply to the weather data.
