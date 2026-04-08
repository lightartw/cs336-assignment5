import json
from typing import Callable, List
from vllm import LLM, SamplingParams
from pathlib import Path

import logging
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.config import (
    QWEN_PATH, TEST_PATH,
    PROMPT_PATH, OUTPUT_DIR
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

OUTPUT_PATH = OUTPUT_DIR / "evaluation_results.json"


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    answers: List[str],
    eval_sampling_params: SamplingParams,
    output_path: Path
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    logger.info(f"Starting generation for {len(prompts)} prompts...")
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    results = []
    for output, expected_answer in zip(outputs, answers):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        
        metrics = reward_fn(generated_text, expected_answer)

        results.append({
            "prompt": prompt,
            "generated_text": generated_text,
            "expected_answer": expected_answer,
            "metrics": metrics
        })
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    logger.info(f"Evaluation complete. Results successfully serialized to {output_path}")


    
def main():
    questions = []
    answers = []
    
    logger.info(f"Reading dataset from {TEST_PATH}...")
    # get prompts
    with TEST_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                questions.append(data["question"])
                answers.append(data["answer"])

    logger.info(f"Reading prompt template from {PROMPT_PATH}...")
    with PROMPT_PATH.open("r", encoding="utf-8") as f:
        prompt_template = f.read()

    prompts = [prompt_template.format(question=q) for q in questions]

    # 初始化 vLLM 及采样参数
    sampling_params = SamplingParams(
        temperature=1.0, 
        top_p=1.0, 
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    # vLLM 初始化需要传入字符串路径
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
    main()