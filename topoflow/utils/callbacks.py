"""Callback to adjust the Gumbel-Softmax temperature using a cosine schedule."""

import math
import lightning.pytorch as pl


class CosineTemp(pl.Callback):
    """Callback to adjust the Gumbel-Softmax temperature using a cosine schedule."""

    def __init__(self, t_start: float, t_min: float, max_epochs: int):
        """Initializes the CosineTemp callback."""
        super().__init__()
        self.t_start = t_start
        self.t_min = t_min
        self.max_epochs = max_epochs

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Set the initial temperature at the start of training."""
        # initialize
        pl_module.gumbel_temp = float(self.t_start)

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Update the temperature using a cosine schedule."""
        epoch = trainer.current_epoch
        # progress in [0,1]
        p = min(1.0, epoch / max(1, self.max_epochs))
        temp = self.t_min + 0.5 * (self.t_start - self.t_min) * (
            1.0 + math.cos(math.pi * p)
        )
        pl_module.gumbel_temp = float(temp)
