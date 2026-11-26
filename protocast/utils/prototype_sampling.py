"""Fixed training-window and patch sampling for ProtoCast."""
import numpy as np


def refit_plan(model, dataset, config):
    """Fixed train windows/patches, drawn in the same RNG order as V1.5.2's fit."""
    settings = config['prototype']
    rng = np.random.default_rng(config['seed'])
    windows = rng.choice(len(dataset), min(settings['init_windows'], len(dataset)), replace=False)
    per_window = int(np.ceil(settings['init_samples'] / len(windows)))
    m = model.backbone
    patches = (m.h // m.p) * (m.w // m.p)
    plan = [(int(w), rng.choice(patches, min(per_window, patches), replace=False)) for w in windows]
    return [(w, p, None) for w, p in plan]
