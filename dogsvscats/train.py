import pytorch_lightning as pl
from dogsvscats.model.lightning import LitDogsVsCats
from dogsvscats import config

litdogsvscats = LitDogsVsCats()

early_stopping = pl.callbacks.EarlyStopping(
    monitor="val_loss", patience=config.EARLY_STOPPING_PATIENCE
)

model_checkpoint = pl.callbacks.ModelCheckpoint(
    dirpath=config.MODEL_DATA_PATH, save_weights_only=True
)

trainer = pl.Trainer(gpus=1, callbacks=[early_stopping, model_checkpoint])
trainer.fit(litdogsvscats)
