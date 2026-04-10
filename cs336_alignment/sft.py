import json
import random
import shutil

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from transformers.tokenization_utils import PreTrainedTokenizer
import wandb
import logging

from cs336_alignment.config import (
    QWEN_PATH, TRAIN_PATH, 
    TEST_PATH, PROMPT_PATH,
    OUTPUT_DIR, SFT_PATH,
    TrainingConfig
)
from cs336_alignment.util import (
    get_response_log_probs,
    sft_microbatch_train_step,
    tokenize_prompt_and_output
)
from cs336_alignment.eval import (
    init_vllm,
    prepare_eval_data,
    run_eval, 
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)



def load_dataset_subset(data_path: Path , num_examples: int|None=None):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    if num_examples is not None:
        data = data[:num_examples]
    return data

def get_micro_batch(
    dataset_subset: list[dict],
    tokenizer: PreTrainedTokenizer,
    micro_batch_size: int
) -> dict[str, torch.Tensor]:
    batch_data = random.sample(dataset_subset, micro_batch_size)

    prompt_strs = [item["question"] for item in batch_data]
    output_strs = [item["answer"] for item in batch_data]
    return tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)

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


def train(config: TrainingConfig):
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
    config.device = policy_device
    
    # 2.wandb
    wandb.init(
        project="qwen-math-sft",
        name=f"sft_samples_{config.num_examples}",
        config=config.model_dump()
    )
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")
    
    # 3.init model or load checkpoint
    start_step = 0
    ckpt_dir = str(QWEN_PATH)

    if SFT_PATH.exists():
        ckpts = sorted(SFT_PATH.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[-1]))
        if ckpts:
            latest_ckpt = ckpts[-1]
            ckpt_dir = str(latest_ckpt)
            start_step = int(latest_ckpt.name.split("-")[-1])
            logger.info(f"Found existing checkpoint. Resuming from {ckpt_dir} at step {start_step}")
        else:
            logger.info(f"No checkpoints found in {SFT_PATH}. Starting from base model.")
    else:
        logger.info("Starting from base model.")

    model = AutoModelForCausalLM.from_pretrained(
        ckpt_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", 
    ).to(config.device)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    # 4.get training dataset (num_examples)
    train_data_subset = load_dataset_subset(TRAIN_PATH, config.num_examples)
    micro_batch_size = config.batch_size // config.grad_accumulate_steps

    # 5.init eval data and vllm
    logger.info("loading evaluation dataset")    
    eval_prompts, eval_answers = prepare_eval_data(TEST_PATH, PROMPT_PATH)
    vllm_engine = init_vllm(
        model_id=str(QWEN_PATH),
        device=vllm_device, 
        seed=42,
        gpu_memory_utilization=vllm_gpu_util
    )

    logger.info(f"Training started on {len(train_data_subset)} examples.")

    # 6.start train
    warmup_steps = int(0.1 * config.max_iters)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=config.max_iters,
        last_epoch=start_step - 1 if start_step > 0 else -1
    )
    model.train()
    for i in range(start_step, config.max_iters):
        optimizer.zero_grad()
        accumlated_loss = 0.0

        for _ in range(config.grad_accumulate_steps):
            # get micro data
            train_batch = get_micro_batch(train_data_subset, tokenizer, micro_batch_size)
            input_ids = train_batch["input_ids"].to(config.device)
            labels = train_batch["labels"].to(config.device)
            response_mask = train_batch["response_mask"].to(config.device)

            # forward
            response_results = get_response_log_probs(
                model=model, 
                input_ids=input_ids, 
                labels=labels, 
                return_token_entropy=True
            )
            log_probs = response_results["log_probs"]
            entropy = response_results["token_entropy"]

            # backward (loss)
            normalize_constant = response_mask.sum(dim=-1).float() 
            normalize_constant = torch.clamp(normalize_constant, min=1e-8) 

            loss, _ = sft_microbatch_train_step(
                policy_log_probs=log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=config.grad_accumulate_steps,
                normalize_constant=normalize_constant
            )
            accumlated_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # logging
        if (i + 1) % config.log_interval == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            avg_entropy = entropy[response_mask].mean().item()

            logger.info(f"Step {i + 1} | Loss: {accumlated_loss:.4f} | LR: {current_lr:.6f}")
            wandb.log({
                "train/loss": accumlated_loss,
                "train/response_entropy": avg_entropy,
                "train/lr": current_lr,
                "step": i + 1
            })

        # eval
        if (i + 1) % config.eval_interval == 0:
            model.eval()
            logger.info(f"Triggering Evaluation at Step {i + 1}")

            with torch.no_grad():
                eval_metrics = run_eval(
                    policy_model=model,
                    vllm_engine=vllm_engine,
                    prompts=eval_prompts, 
                    answers=eval_answers, 
                    output_dir=OUTPUT_DIR, 
                    step= i + 1
                )
            wandb.log({
                "eval/format_count": eval_metrics["format_correct_count"],
                "eval/answer_count": eval_metrics["answer_correct_count"],
                "eval/accuracy": eval_metrics["accuracy"],
                "step": i + 1
            })

            model.train()

        # save 4 time
        if (i + 1) % int(config.max_iters // 4) == 0:
            ckpt_dir = save_checkpoint(model, tokenizer, SFT_PATH, step=i+1)