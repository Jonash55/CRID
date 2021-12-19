import torch
import numpy as np
import torch.nn.functional as F

from typing import Any


def images_to_probs(net, images: Any):
    """
    Convert output probabilities to predicted class

    Parameters
    ----------
    net: ConvNet
        Trained model
    images: Any
        Images from dataset
    Returns
    -------
    predictions,
    """
    output = net(images)

    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]
