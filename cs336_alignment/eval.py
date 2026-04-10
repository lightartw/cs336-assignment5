import json
from typing import Callable, List, Tuple
from unittest.mock import patch
from transformers.modeling_utils import PreTrainedModel
from vllm import LLM, SamplingParams
from vllm.model_executor.utils import set_random_seed as vllm_set_random_seed
import torch

from pathlib import Path
import logging
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def prepare_eval_data(test_path: Path, prompt_path: Path) -> Tuple[List[str], List[str]]:
    questions = []
    answers = []
    
    with test_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                questions.append(data["question"])
                answers.append(data["answer"])

    with prompt_path.open("r", encoding="utf-8") as f:
        prompt_template = f.read()

    prompts = [prompt_template.format(question=q) for q in questions]
    
    return prompts, answers

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)
    
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
            "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
            return_value=None
        )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    answers: List[str],
    eval_sampling_params: SamplingParams,
    output_path: Path
) -> dict[str, float]:
    """
    Evaluate a language model, compute evaluation metrics, serialize results, 
    and return a dictionary of detailed metrics.
    """
    logger.info(f"Starting generation for {len(prompts)} prompts...")
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    results = []
    
    format_correct_count = 0
    answer_correct_count = 0
    total_reward_sum = 0.0
    num_samples = len(prompts)

    for output, expected_answer in zip(outputs, answers):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        
        metrics = reward_fn(generated_text, expected_answer)
        
        if metrics.get("format_reward", 0.0) == 1.0:
            format_correct_count += 1
        if metrics.get("answer_reward", 0.0) == 1.0:
            answer_correct_count += 1
        total_reward_sum += metrics.get("reward", 0.0)

        results.append({
            "prompt": prompt,
            "generated_text": generated_text,
            "expected_answer": expected_answer,
            "metrics": metrics
        })
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    metrics_summary = {
        "format_correct_count": format_correct_count,
        "answer_correct_count": answer_correct_count,
        "total_samples": num_samples,
        "accuracy": (total_reward_sum / num_samples) * 100 if num_samples > 0 else 0.0,
    }
    
    return metrics_summary

def run_eval(
    policy_model: PreTrainedModel,
    vllm_engine: LLM,
    prompts: List[str],
    answers: List[str],
    output_dir: Path,  
    step: int
) -> dict[str, float]:
    logger.info("Syncing updated policy weights into vLLM engine...")
    load_policy_into_vllm_instance(policy_model, vllm_engine)

    sampling_params = SamplingParams(
        temperature=0.0, 
        top_p=1.0, 
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    step_output_path = output_dir / f"evaluation_results_step_{step}.json"

    eval_metrics = evaluate_vllm(
        vllm_model=vllm_engine,
        reward_fn=r1_zero_reward_fn, 
        prompts=prompts,
        answers=answers,
        eval_sampling_params=sampling_params,
        output_path=step_output_path
    )
    
    return eval_metrics

# ==================== test ================
from cs336_alignment.config import (
    QWEN_PATH, TEST_PATH,
    PROMPT_PATH, OUTPUT_DIR
)

OUTPUT_PATH = OUTPUT_DIR / "evaluation_results.json"

def eval():
    
    logger.info(f"Reading dataset from {TEST_PATH}...")
    prompts, answers = prepare_eval_data(TEST_PATH, PROMPT_PATH)

    # 初始化 vLLM 及采样参数
    sampling_params = SamplingParams(
        temperature=1.0, 
        top_p=1.0, 
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    # vLLM 
    logger.info(f"Loading model from {QWEN_PATH}...")
    llm = LLM(model=str(QWEN_PATH))

    # 调用评估函数
    evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        answers=answers,
        eval_sampling_params=sampling_params,
        output_path=OUTPUT_PATH
    )

if __name__ == "__main__":
    eval()