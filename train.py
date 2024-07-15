import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import optim
from torch.optim import Adam
from torch_ema import ExponentialMovingAverage as EMA

import diffusers  # noqa: F401
from data.umbra import UmbraDataModule


cfg = OmegaConf.load("config.yaml")
torch.set_float32_matmul_precision("medium")


class Diffusion(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg
        self.model = eval(self.cfg.model._target_)(**self.cfg.model.params)
        self.train_scheduler = eval(self.cfg.noise._target_)(**self.cfg.noise.params)
        self.infer_scheduler = eval(self.cfg.noise._target_)(**self.cfg.noise.params)

        self.ema = (
            EMA(self.model.parameters(), decay=self.cfg.hp.training.ema_decay)
            if self.cfg.hp.training.ema_decay != -1
            else None
        )

        # self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        clean_images = batch
        noise = torch.randn_like(clean_images)
        timesteps = torch.randint(
            0,
            self.train_scheduler.config.num_train_timesteps,
            size=(clean_images.size(0),),
            device=self.device,
        ).long()

        noisy_images = self.train_scheduler.add_noise(clean_images, noise, timesteps)
        model_output = self.model(noisy_images, timesteps).sample

        loss = F.mse_loss(model_output, noise)
        log_key = f'{"train" if self.training else "val"}/simple_loss'
        self.log_dict(
            {
                log_key: loss,
            },
            prog_bar=True,
            sync_dist=True,
            on_step=self.training,
            on_epoch=not self.training,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=cfg.hp.training.lr)
        sched = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch", "frequency": 1},
        }


def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    system = Diffusion(cfg)
    datamodule = UmbraDataModule(cfg.dm)

    os.makedirs(cfg.dm.output_dir, exist_ok=True)

    cb_ckpt = pl.callbacks.ModelCheckpoint(
        dirpath=cfg.dm.output_dir,
        filename=cfg.dm.ckpt_filename,
        monitor="val/simple_loss",
        save_top_k=1,
        mode="min",
        save_last=True,
    )

    trainer = pl.Trainer(
        logger=eval(cfg.log._target_)(**cfg.log.params),
        callbacks=[
            pl.callbacks.LearningRateMonitor(
                "epoch", log_momentum=True, log_weight_decay=True
            ),
            pl.callbacks.RichProgressBar(),
            cb_ckpt,
        ],
        **cfg.hp.pl_trainer,
    )
    trainer.fit(system, datamodule=datamodule, ckpt_path=cfg.hp.resume_from_checkpoint)


if __name__ == "__main__":
    main(cfg)
