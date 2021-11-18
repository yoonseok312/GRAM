from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_noam_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    only_warmup: bool = False,
    interval: str = "step",
):
    """
    Noam learning rate scheduler for pytorch-lightning.
    To use this method appropriately, define `configure_optimizers` of
    pytorch lightning module as follows:
    ```
    def configure_optimizers(self):
        optimizer = ...
        if self.hp.scheduler == "noam":
            scheduler = get_noam_scheduler(optimizer, self.hp.warmup_steps)
            return [optimizer], [scheduler]
    ```
    Inputs:
        interval: bool. Default value is "False".
                  if False, it will return original noam_scheduler.
                  else, if step > warmup, it will return 1.
        interval: "step" or "epoch". Default value is "step". epoch-wise
                  Noam scheduling will be rarely used. (may not be used at all)
    """

    def noam(step: int):
        """
        Learning rate scale
        Here step will be epoch or global_step, starts from 0
        """
        scale = warmup_steps ** 0.5 * min(
            (step + 1) ** (-0.5), (step + 1) * warmup_steps ** (-1.5)
        )
        if only_warmup and step >= warmup_steps:
            scale = 1

        return scale

    scheduler = LambdaLR(optimizer=optimizer, lr_lambda=noam)
    scheduler = {
        "scheduler": scheduler,
        "interval": interval,
        "frequency": 1,
    }
    return scheduler
