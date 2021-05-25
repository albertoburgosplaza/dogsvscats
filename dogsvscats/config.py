from pathlib import Path
import torch

PROJECT_PATH = Path("/home/alberto/workspace/dogsvscats")
DATA_PATH = PROJECT_PATH / "data"
TRAIN_DATA_PATH = DATA_PATH / "train"
TEST_DATA_PATH = DATA_PATH / "test1"
MODEL_DATA_PATH = DATA_PATH / "model"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
SCHEDULER_PATIENCE = 5
LR = 0.01
BS = 128
NW = 4
SAMPLE_SIZE = 1

CLASSES = {0: "cat", 1: "dog"}
INV_CLASSES = {v: k for k, v in CLASSES.items()}

MODEL_NAME = "resnet18_2021-01-26"
# CHECKPOINT_PATH = "/home/alberto/workspace/dogsvscats/mlruns/1/1647c6b15f684db7b17798d28d168ee2/artifacts/restored_model_checkpoint/epoch=26-step=4751.ckpt"
CHECKPOINT_PATH = None
# MODEL_PATH = "/home/alberto/workspace/dogsvscats/data/model/0/60bd1d1908884479b9a9d0dfa7e869f2/artifacts/model/data/model.pth"
