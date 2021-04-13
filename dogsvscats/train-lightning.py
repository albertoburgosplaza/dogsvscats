import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from dogsvscats.model.model import get_model
from dogsvscats.data.dataset import get_datasets
from dogsvscats.data.transforms import train_tfs, val_tfs
from dogsvscats import config


class LitDogsVsCats(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = get_model()

    def forward(self, x):
        return self.model(x)

    def prepare_data(self):
        datasets = get_datasets(
            sample_size=config.SAMPLE_SIZE, train_tfs=train_tfs, val_tfs=val_tfs
        )
        self.train_dataset = datasets["train"]
        self.valid_dataset = datasets["val"]

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=config.BS,
            shuffle=True,
            num_workers=config.NW,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=config.BS,
            shuffle=False,
            num_workers=config.NW,
        )

        return valid_loader

    def configure_optimizers(self):
        optimizer = optim.SGD(self.model.parameters(), lr=config.LR, momentum=0.9)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=config.SCHEDULER_PATIENCE, verbose=True
        )

        return {"optimizer": optimizer, "scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        images, labels = batch

        outputs = self.model(images)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)

        self.log("train_loss", loss)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        outputs = self.model(images)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)

        self.log("val_loss", loss)

        return {"loss": loss}


litdogsvscats = LitDogsVsCats()

early_stopping = pl.callbacks.EarlyStopping(
    monitor="val_loss", patience=config.EARLY_STOPPING_PATIENCE, verbose=True
)
# model_checkpoint = pl.callbacks.ModelCheckpoint(**cfg.callbacks.model_checkpoint.params)

# tb_logger = pl.lo TensorBoardLogger(save_dir=cfg.general.save_dir)

trainer = pl.Trainer(gpus=1, callbacks=[early_stopping])
trainer.fit(litdogsvscats)
