import torch
from torchvision import models
import torch.nn as nn
from torch.utils import model_zoo
from collections import namedtuple
from dogsvscats.model.utils import get_state_dict_from_lightning_checkpoint


MODEL = namedtuple("model", ["url"])

MODELS = {
    "resnet18_2021-01-26": MODEL(
        url="https://github.com/albertoburgosplaza/dogsvscats/releases/download/modelweights-v0.0.1/resnet18_2021-01-26.zip"  # noqa: E501
    )
}


def get_model(model_path=None, checkpoint_path=None, model_name=None):
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    state_dict = None

    if model_path:
        model = torch.load(model_path)
        return model
    elif checkpoint_path:
        state_dict = get_state_dict_from_lightning_checkpoint(checkpoint_path)
    elif model_name:
        state_dict = model_zoo.load_url(
            MODELS[model_name].url, progress=True, map_location="cpu"
        )

    if state_dict:
        model.load_state_dict(state_dict)

    return model
