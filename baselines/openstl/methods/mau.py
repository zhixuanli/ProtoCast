import torch
from .base_method import Base_method
from openstl.models import MAU_Model
from openstl.utils import schedule_sampling


class MAU(Base_method):
    r"""MAU

    Implementation of `MAU: A Motion-Aware Unit for Video Prediction and Beyond
    <https://openreview.net/forum?id=qwtfY-3ibt7>`_.

    """

    def __init__(self, **args):
        super().__init__(**args)
        self.eta = 1.0

    def _build_model(self, **args):
        num_hidden = [int(x) for x in self.hparams.num_hidden.split(',')]
        num_layers = len(num_hidden)
        return MAU_Model(num_layers, num_hidden, self.hparams)
    
    def forward(self, batch_x, batch_y=None, **kwargs):
        _, img_channel, img_height, img_width = self.hparams.in_shape
        # Autoregressive inference needs only observations, never future targets.
        future = batch_x.new_zeros((batch_x.shape[0], self.hparams.aft_seq_length,
                                   *batch_x.shape[2:]))
        test_ims = torch.cat([batch_x, future], dim=1).permute(0, 1, 3, 4, 2).contiguous()
        real_input_flag = torch.zeros(
            (batch_x.shape[0], self.hparams.aft_seq_length - 1, 1, 1, 1),
            device=self.device)
        img_gen, _ = self.model(test_ims, real_input_flag, return_loss=False)
        pred_y = img_gen[:, -self.hparams.aft_seq_length:, :]
        return pred_y
    
    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
        self.eta, real_input_flag = schedule_sampling(
            self.eta, self.global_step, ims.shape[0], self.hparams,
            update_eta=(batch_idx % getattr(self._trainer, 'accumulate_grad_batches', 1) == 0))
        img_gen, loss = self.model(ims, real_input_flag)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
