import torch

from algorithms.value_loss import compute_value_loss


predicted_values = torch.tensor([
    10.0,
    8.0,
    6.0
])

returns = torch.tensor([
    12.0,
    9.0,
    7.0
])


loss = compute_value_loss(
    predicted_values,
    returns
)

print(loss)