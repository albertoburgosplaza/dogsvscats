import mlflow
import pytorch_lightning as pl
from dogsvscats.model.lightning import LitDogsVsCats
from dogsvscats import config

mlflow.set_experiment("training")
mlflow.pytorch.autolog()

litdogsvscats = LitDogsVsCats()

early_stopping = pl.callbacks.EarlyStopping(
    monitor="val_loss", patience=config.EARLY_STOPPING_PATIENCE
)

trainer = pl.Trainer(
    gpus=1,
    callbacks=[early_stopping],
    logger=None,
    resume_from_checkpoint=config.CHECKPOINT_PATH,
)

with mlflow.start_run() as run:
    trainer.fit(litdogsvscats)
