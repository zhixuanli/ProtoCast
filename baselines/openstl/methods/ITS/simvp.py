import torch
from openstl.models.ITS.med_ciatt_mutinums import SimVP_Model



from openstl.methods.base_method import Base_method

class SimVP(Base_method):
    r"""SimVP

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_

    """

    def __init__(self, **args):
        super().__init__(**args)
        self.test_num = 0

    def _build_model(self, **args):
        return SimVP_Model(**args)

    def forward(self, batch_x, batch_y=None, **kwargs):
        if batch_y is None or not self.training:
            return self.model.sample(batch_x)
        return self.model(batch_x, batch_y, **kwargs)

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        accumulation = getattr(self._trainer, 'accumulate_grad_batches', 1)
        _, loss, _, _, _ = self(
            batch_x, batch_y, update_encoder_decoder_ema=(batch_idx % accumulation == 0))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_after_backward(self):
        # Same pre-optimizer parameters as Algorithm 2, after the old frozen
        # translator weights have finished propagating gradients to the encoder.
        if self.training and getattr(self._trainer, 'accumulate_grad_batches', 1) == 1:
            self.model._momentum_update_key_translator()

    def on_before_optimizer_step(self, optimizer):
        # Both microbatches need the same frozen translator parameters while
        # propagating gradients. Update once after all backwards, before Adam.
        if self.training and getattr(self._trainer, 'accumulate_grad_batches', 1) > 1:
            self.model._momentum_update_key_translator()

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x)
        loss = self.criterion(pred_y, batch_y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        self._record_validation_batch(batch_x, pred_y, batch_y)
        return loss

    def test_step(self, batch, batch_idx):
        self.test_num = self.test_num + 1
        batch_x, batch_y = batch
        pred_y = self(batch_x)
        return self._record_test_batch(batch_x, pred_y, batch_y)
