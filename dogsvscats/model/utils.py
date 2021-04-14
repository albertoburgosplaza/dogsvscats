from typing import Dict
import torch


def get_state_dict_from_lightning_checkpoint(checkpoint_path) -> Dict:

    state_dict = torch.load(checkpoint_path)["state_dict"]

    new_state_dict = {}

    for k in state_dict.keys():
        new_k = k.replace("model.", "")
        new_state_dict[new_k] = state_dict[k]

    return new_state_dict
