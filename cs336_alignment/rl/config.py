from typing import Literal
from pydantic import BaseModel

class GRPOConfig(BaseModel):
    n_grpo_steps: int=200
    learning_rate: float = 1e-5

    rollout_batch_size: int = 256
    group_size: int = 8
    train_batch_size: int = 256 # On-policy
    gradient_accumulation_steps: int = 128 # microbatch size is 2, will fit on H100

    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4 # As in Expiter, disallow empty string responses
    sampling_max_tokens: int = 1024
    advantage_eps: float = 1e-6
    
    loss_type: Literal[
        "no_baseline",
        "reinforce_with_baseline",
        "grpo_clip",
        ] = "reinforce_with_baseline"
    epochs_per_rollout_batch: int = 1 # On-policy
    use_std_normalization: bool = True

    num_examples: int = 512
    eval_interval: int = 20
    log_interval: int = 5

    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"