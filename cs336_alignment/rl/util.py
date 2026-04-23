from typing import Callable, Literal
import torch
from cs336_alignment.util import masked_mean

def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    
    metadate: dict[str, float] = None
    rollout_batch_size = len(repeated_ground_truths)
    assert (rollout_batch_size % group_size == 0)

    # get all reward
    repeated_reward: list[float] = []
    for response, truth in zip(rollout_responses, repeated_ground_truths):
        reward_dict = reward_fn(response, truth)
        repeated_reward.append(reward_dict.get("reward", 0.0))

    raw_rewards = torch.tensor(repeated_reward, dtype=torch.float32)

    # cal advantages
    grouped_rewards = raw_rewards.view(-1, group_size)      # [n_prompts_per_rollout_batch, group_size]
    group_means = grouped_rewards.mean(dim=1, keepdim=True) # [n_prompts_per_rollout_batch, 1]

    advantages = grouped_rewards - group_means

    if normalize_by_std:
        group_stds = grouped_rewards.std(dim=1, keepdim=True, unbiased=True)
        advantages = advantages / (group_stds + advantage_eps)

    advantages = advantages.view(-1) # [rollout_batch_size, ]

    return advantages, raw_rewards, metadate

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor: 
    """
    Args:
        raw_rewards_or_advantages: torch.Tensor, shape (batch_size, 1)
        policy_log_probs: torch.Tensor, shape (batch_size, sequence_length)

    Returns
        torch.Tensor, shape (batch_size, sequence_length), per-token policy-gradient loss
    """
    return  -(policy_log_probs * raw_rewards_or_advantages)


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Args:
        advantages: torch.Tensor Shape (batch_size, 1)
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length)
        old_log_probs: torch.Tensor Shape (batch_size, sequence_length)
        cliprange: float Clip parameter ε (e.g. 0.2).
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss torch.Tensor of shape (batch_size, sequence_length)
            metadata dict containing whatever you want to log. 
    """
    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    ratio_clipped = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    
    pg_unclipped = ratio * advantages
    pg_clipped = ratio_clipped * advantages

    # prepare metadata
    clipped_mask = (pg_clipped < pg_unclipped)
    metadata = {
        "ratio": ratio.detach(),
        "ratio_clipped": ratio_clipped.detach(),
        "clipped_mask": clipped_mask.detach(),
        "clipfrac": clipped_mask.float().mean().detach(),
    }
    return -torch.minimum(pg_unclipped, pg_clipped), metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
    if loss_type == "no_baseline":
        assert raw_rewards is not None, "raw_rewards is required for no_baseline"
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        return loss, None
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, "advantages is required for reinforce_with_baseline"
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        return loss, None
    elif loss_type == "grpo_clip":
        assert advantages is not None, "advantages is required for grpo_clip"
        assert old_log_probs is not None, "old_log_probs is required for grpo_clip"
        assert cliprange is not None, "cliprange is required for grpo_clip"
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)

    raise ValueError(f"Unknown loss_type: {loss_type}")

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs, loss_type,
        raw_rewards, advantages,
        old_log_probs, cliprange)
    masked_loss = masked_mean(loss, response_mask, dim=-1) # [batch_size, ]
    loss = masked_loss.mean()
    loss = loss / gradient_accumulation_steps
    loss.backward()

    return loss, metadata