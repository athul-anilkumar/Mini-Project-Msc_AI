import torch


def compute_actor_loss(
    new_log_probs,
    old_log_probs,
    advantages,
    clip_eps=0.2
):

    ratio = torch.exp(
        new_log_probs - old_log_probs
    )

    surrogate1 = ratio * advantages

    surrogate2 = torch.clamp(
        ratio,
        1 - clip_eps,
        1 + clip_eps
    ) * advantages

    loss = -torch.min(
        surrogate1,
        surrogate2
    ).mean()

    return loss