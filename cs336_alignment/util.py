import json
from pathlib import Path
import shutil
from typing import Callable, List

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def tokenize_prompt_and_output(
        prompt_strs: list[str], 
        output_strs: list[str], 
        tokenizer: PreTrainedTokenizer
) -> dict[str, torch.Tensor] :
    max_prompt_and_output_len = 0
    result = {"input_ids": [], "labels": [], "response_mask": []}

    for prompt, output in zip(prompt_strs, output_strs):
        prompt_token = tokenizer(prompt)["input_ids"]
        output_token = tokenizer(output)["input_ids"]
        full = prompt_token + output_token
        max_prompt_and_output_len = max(len(full), max_prompt_and_output_len)            

        result["input_ids"].append(full)
        
        prompt_mask_len = len(prompt_token) - 1
        output_mask_len = len(output_token)
        mask = [False] * prompt_mask_len + [True] * output_mask_len
        result["response_mask"].append(mask)

    # padding
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    for i in range(len(result["input_ids"])):
        full = result["input_ids"][i]
        pad_len = max_prompt_and_output_len - len(full)
        full_pad = full + [pad_id] * pad_len
        
        result["input_ids"][i] = full_pad[:-1]
        result["labels"].append(full_pad[1:])
        result["response_mask"][i].extend([False] * pad_len)

    return {
        "input_ids": torch.tensor(result["input_ids"], dtype=torch.long),
        "labels": torch.tensor(result["labels"], dtype=torch.long),
        "response_mask": torch.tensor(result["response_mask"], dtype=torch.bool)
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Args:
        logits: torch.Tensor  (batch_size, sequence_length, vocab_size)

    Returns:
        torch.Tensor (batch_size, sequence_length)
    """
    log_p = torch.log_softmax(logits, dim=-1)
    p = torch.exp(log_p)

    return - torch.sum(p * log_p, dim=-1)

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits

    log_prob = torch.log_softmax(logits, dim=-1)

    sel_log_prob = torch.gather(log_prob, dim=-1, index=labels.unsqueeze(-1))
    log_prob = sel_log_prob.squeeze(-1)

    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        return {
            "log_probs": log_prob,
            "token_entropy": token_entropy
        }
    return {
        "log_probs": log_prob,
        "token_entropy": None
    }

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    masked_tensor = torch.where(mask, tensor, torch.zeros_like(tensor))
    sum_tensor = torch.sum(masked_tensor, dim=dim)
    return sum_tensor / normalize_constant

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    masked_norm = masked_normalize(policy_log_probs, response_mask, normalize_constant, dim=-1)
    loss = -masked_norm.mean()
    loss = loss / gradient_accumulation_steps
    loss.backward()

    return loss, {}

def log_generations(
    model: PreTrainedModel,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompt: List[str],
    label: List[str],
) -> None:
    raise NotADirectoryError


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    masked_tensor = torch.where(mask, tensor, torch.zeros_like(tensor))
    sum_tensor = masked_tensor.sum(dim=dim)
    
    count = mask.sum(dim=dim)
    return sum_tensor / count


# training util

def load_dataset_subset(data_path: Path , num_examples: int|None=None):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    if num_examples is not None:
        data = data[:num_examples]
    return data

def save_checkpoint(model, tokenizer, sft_path: Path, step: int, max_to_keep: int = 2):
    sft_path.mkdir(parents=True, exist_ok=True)
    ckpt_dir = sft_path / f"checkpoint-{step}"
    
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    logger.info(f"Checkpoint saved to {ckpt_dir}")
    
    ckpts = sorted(sft_path.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[-1]))
    for old_ckpt in ckpts[:-max_to_keep]:
        shutil.rmtree(old_ckpt)

    return ckpt_dir