import torch
import torch.nn.functional as F


def compute_value_loss(
    predicted_values,
    returns
):

    loss = F.mse_loss(
        predicted_values,
        returns
    )

    return loss