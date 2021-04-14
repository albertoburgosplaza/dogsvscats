import torch
import torch.nn.functional as F
from dogsvscats import config
from dogsvscats.data.dataset import pil_loader
from dogsvscats.data.transforms import default_tfs
from dogsvscats.data.utils import imshow
from dogsvscats.model.model import get_model


def predict_image(
    image_path,
    model=get_model(
        checkpoint_path=config.CHECKPOINT_PATH, model_name=config.MODEL_NAME
    ),
    tfs=default_tfs,
    device="cpu",
    show_image=True,
):
    model = model.to(device)
    model.eval()

    img = pil_loader(image_path)
    transformed_img = tfs(img)
    input = transformed_img.unsqueeze(0)
    input = input.to(device)
    output = model(input)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)
    pred_label_idx.squeeze_()
    predicted_label = config.CLASSES[pred_label_idx.item()]

    if show_image:
        imshow(
            transformed_img,
            f"Predicted: {predicted_label} ({prediction_score.squeeze().item()})",
        )

    return {
        "label": predicted_label,
        "idx_label": pred_label_idx,
        "score": prediction_score.squeeze().item(),
    }
