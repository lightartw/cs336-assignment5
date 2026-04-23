from pathlib import Path
from pydantic import BaseModel

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# data
DATA_DIR = PROJECT_ROOT / "data" / "gsm8k"
TEST_PATH = DATA_DIR / "test.jsonl"
TRAIN_PATH = DATA_DIR / "train.jsonl"
PROMPT_PATH = CURRENT_DIR / "prompts" / "r1_zero.prompt"

# model
MODEL_DIR = PROJECT_ROOT / "models"
QWEN_PATH = MODEL_DIR / "Qwen2.5-0.5B"       # "Qwen2.5-Math-1.5B"
SFT_PATH = MODEL_DIR /  "Qwen2.5-0.5B-sft"   # "Qwen2.5-Math-1.5B-sft"

# output
OUTPUT_DIR = PROJECT_ROOT / "results"

class TrainingConfig(BaseModel):
    lr: float
    batch_size: int
    max_iters: int
    grad_accumulate_steps: int # micro_batch_size = batch_size / grad_accumulate_steps
    device: str="cuda"

    num_examples: int # for dateset
    eval_interval: int
    log_interval: int