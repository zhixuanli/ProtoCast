"""ProtoCast with gated post-adapter variable exchange."""
import torch
from torch import nn

from protocast.models._shared import (
    GroupedVariableBackbone, _GroupedPrototypeModel, _apply_independent_batches,
)
from protocast.modules.blocks import GatedBlock
from protocast.utils.checkpoint import seed_all


class ProtoCast(_GroupedPrototypeModel):
    """Keep independent banks/adapters and exchange their corrected features."""

    def __init__(self, backbone, settings):
        super().__init__(backbone, settings)
        self.post_adapter_attention = GatedBlock(
            backbone.dim, backbone.config['heads'], backbone.config['dropout'])
        # The new stage starts as an identity. Its ordinary random weights give
        # the scalar a nonzero learning signal before the stage itself trains.
        self.post_adapter_gate = nn.Parameter(torch.tensor(
            float(backbone.config.get('post_adapter_fusion_gate_init', 0.0))))


    def fuse_adapted(self, h):
        """Attend over four variables independently at each lead and patch."""
        b, t, g, n, d = h.shape
        sequences = h.permute(0, 1, 3, 2, 4).reshape(b * t * n, g, d)
        fused = _apply_independent_batches(self.post_adapter_attention, sequences)
        fused = fused.reshape(b, t, n, g, d).permute(0, 1, 3, 2, 4)
        return h + self.post_adapter_gate.to(h.dtype) * (fused - h)

    def forward(self, x):
        h, descriptors = self.prefix(x)
        adapted = []
        for index, name in enumerate(self.variable_groups):
            group = h[:, :, index]
            route = self.banks[name](descriptors[name].detach())
            delta = self.adapters[name](group, route)
            adapted.append(group + delta)
        return self.backbone.decode(self.fuse_adapted(torch.stack(adapted, dim=2)), x)


def build_protocast(config):
    data = config['data']
    variables = data['variables']
    model_config = dict(config['model'], height=data['height'], width=data['width'],
                        input_steps=data['input_steps'], output_steps=data['output_steps'],
                        channels=len(variables), target_indices=list(range(len(variables))))
    groups = {name: [variables.index(variable) for variable in members]
              for name, members in config['prototype']['variable_groups'].items()}
    seed_all(config['seed'])
    backbone = GroupedVariableBackbone(model_config, groups)
    seed_all(config['seed'] + 1)
    return ProtoCast(backbone, config['prototype'])
