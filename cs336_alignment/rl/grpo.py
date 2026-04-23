import json
import random
import shutil

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams
import wandb
import logging

from cs336_alignment.config import (
    QWEN_PATH, SFT_PATH, TRAIN_PATH, 
    TEST_PATH, PROMPT_PATH,
    OUTPUT_DIR, GRPO_PATH
)
from cs336_alignment.rl.config import GRPOConfig
from cs336_alignment.eval import (
    init_vllm,
    load_policy_into_vllm_instance,
    prepare_eval_data,
    run_eval, 
)
from cs336_alignment.rl.util import compute_group_normalized_rewards, grpo_microbatch_train_step, prepare_grpo_inputs
from cs336_alignment.util import get_response_log_probs, load_dataset_subset, save_checkpoint, tokenize_prompt_and_output
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def sample_questions(dataset, num_samples: int) -> list[dict]:
    if num_samples > len(dataset):
        logger.warning(f"Requested {num_samples} samples, but dataset only has {len(dataset)}. Using full dataset.")
        return list(dataset)
    return random.sample(list(dataset), num_samples)

def rollout(
    batch_data: list[dict], 
    vllm_engine, 
    group_size: int, 
    sampling_params
) -> tuple[list[str], list[str], list[str]]:
    """
    Args:
        batch_data: include "question" and "answer"。
        vllm_engine
        group_size
        sampling_params
    """
    questions = [d["question"] for d in batch_data]
    ground_truths = [d.get("answer", "") for d in batch_data]

    repeated_questions = [q for q in questions for _ in range(group_size)]
    repeated_ground_truths = [gt for gt in ground_truths for _ in range(group_size)]

    outputs = vllm_engine.generate(repeated_questions, sampling_params)
    rollout_responses = [out.outputs[0].text for out in outputs]

    return rollout_responses, repeated_ground_truths, repeated_questions


def train(config: GRPOConfig):
    # 1.detect device
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        policy_device = "cuda:0"
        vllm_device = "cuda:1"
        vllm_gpu_util = 0.85
        logger.info("Detected >=2 GPUs. Policy on cuda:0, vLLM on cuda:1.")
    else:
        policy_device = "cuda:0"
        vllm_device = "cuda:0"
        vllm_gpu_util = 0.6
        logger.info("Detected 1 GPU. Policy and vLLM will share cuda:0.")
    device = policy_device
    dtype = getattr(torch, config.dtype)
    
    # 2.wandb
    wandb.init(
        project="qwen-0.5B-grpo",
        name=f"sft_samples_{config.num_examples}",
        config=config.model_dump()
    )
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")
    
    # 3.init model or load checkpoint
    start_step = 0
    ckpt_dir = str(SFT_PATH)

    if GRPO_PATH.exists():
        ckpts = sorted(GRPO_PATH.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[-1]))
        if ckpts:
            latest_ckpt = ckpts[-1]
            ckpt_dir = str(latest_ckpt)
            start_step = int(latest_ckpt.name.split("-")[-1])
            logger.info(f"Found existing checkpoint. Resuming from {ckpt_dir} at step {start_step}")
        else:
            logger.info(f"No checkpoints found in {GRPO_PATH}. Starting from base model.")
    else:
        logger.info("Starting from base model.")

    model = AutoModelForCausalLM.from_pretrained(
        ckpt_dir,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2", 
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    # 4.get training dataset (num_examples)
    train_data_subset = load_dataset_subset(TRAIN_PATH, config.num_examples)
    assert config.train_batch_size % config.gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = config.train_batch_size // config.gradient_accumulation_steps

    # 5.init eval data and vllm
    logger.info("loading evaluation dataset")    
    eval_prompts, eval_answers = prepare_eval_data(TEST_PATH, PROMPT_PATH)
    vllm_engine = init_vllm(
        model_id=str(QWEN_PATH),
        device=vllm_device, 
        seed=42,
        gpu_memory_utilization=vllm_gpu_util,
        dtype=dtype
    )

    rollout_sample_params = SamplingParams(
        temperature=config.sampling_temperature,
        max_tokens=config.sampling_max_tokens,
        min_tokens=config.sampling_min_tokens,
        stop=["</answer>"], # R1_ZERO_PROMPT
        include_stop_str_in_output=True
    )

    logger.info(f"Training started on {len(train_data_subset)} examples.")

    # 6.start train
    model.train()
    for step in range(start_step, config.n_grpo_steps):
        # --- PHASE 1: ROLLOUT ---
        load_policy_into_vllm_instance(model, vllm_engine) # sync params

        assert config.rollout_batch_size % config.group_size == 0, (
            "rollout_batch_size must be divisible by group_size"
        )
        num_prompts = config.rollout_batch_size // config.group_size
        batch_data = sample_questions(train_data_subset, num_prompts)

        rollout_responses, repeated_ground_truths, repeated_questions = rollout(
            batch_data=batch_data,
            vllm_engine=vllm_engine,
            group_size=config.group_size,
            sampling_params=rollout_sample_params
        )

        # compute reawrds and advantages
        advantages, raw_rewards, _ = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=config.group_size,
            advantage_eps=config.advantage_eps,
            normalize_by_std=config.use_std_normalization
        )
        advantages = advantages.to(device)

        # get log_probs
        train_inputs = tokenize_prompt_and_output(
            prompt_strs=repeated_questions,
            output_strs=rollout_responses,
            tokenizer=tokenizer
        )
        input_ids = train_inputs["input_ids"].to(device)
        labels = train_inputs["labels"].to(device)
        res_mask = train_inputs["response_mask"].to(device)
        with torch.no_grad():
            old_res = get_response_log_probs(model, input_ids, labels, return_token_entropy=True)
            old_log_probs = old_res["log_probs"].detach()


        # --- PHASE 2: OPTIMIZATION ---
        for _ in range(config.epochs_per_rollout_batch):
            optimizer.zero_grad()
            
            num_microbatches = config.rollout_batch_size // micro_train_batch_size
            total_loss = 0.0
            
            for m_idx in range(num_microbatches):
                start = m_idx * micro_train_batch_size
                end = start + micro_train_batch_size
                
                m_input_ids = input_ids[start:end]
                m_mask = res_mask[start:end]
                m_advantages = advantages[start:end]
                m_old_log_probs = old_log_probs[start:end]
                
                # Forward
                current_res = get_response_log_probs(model, m_input_ids, m_mask)
                current_log_probs = current_res["log_probs"]
                token_entropy = current_res["token_entropy"]
                
                # Backward
                loss, _ = grpo_microbatch_train_step(
                    policy_log_probs=current_log_probs,
                    response_mask=m_mask,
                    gradient_accumulation_steps=config.gradient_accumulation_steps,
                    loss_type=config.loss_type,
                    advantages=m_advantages,
                    old_log_probs=m_old_log_probs,
                    cliprange=0.2 
                )
                total_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()


        # logging
        if (step + 1) % config.log_interval == 0:
            wandb.log({
                "train/loss": total_loss,
                "train/avg_reward": raw_rewards.mean().item(),
                "train/entropy": token_entropy[res_mask[:micro_train_batch_size]].mean().item(),
                "train_step": step + 1
            })

        # evaluate
        if (step + 1) % config.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                eval_metrics = run_eval(
                    policy_model=model,
                    vllm_engine=vllm_engine,
                    prompts=eval_prompts, 
                    answers=eval_answers, 
                    output_dir=OUTPUT_DIR,
                    step=step + 1
                )
            wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=step+1)
            model.train()

        if (step + 1) % (config.n_grpo_steps // 4) == 0:
            save_checkpoint(model, tokenizer, GRPO_PATH, step=step+1)