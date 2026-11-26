import sys
from pathlib import Path

from .simvp import SimVP


class PredFormer(SimVP):
    def _build_model(self, model_config, **kwargs):
        cache = Path(__file__).resolve().parents[2] / ".cache" / "predformer"
        if not (cache / "PredFormer_Quadruplet_TSST.py").is_file():
            raise FileNotFoundError("Run bash baselines/prepare_predformer.sh first.")
        sys.path.insert(0, str(cache))
        from PredFormer_Quadruplet_TSST import PredFormer_Model
        return PredFormer_Model(model_config=model_config)
