from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

DATA_DIR = PROJECT_ROOT / "data" / "gsm8k"
TEST_PATH = DATA_DIR / "test.jsonl"
TRAIN_PATH = DATA_DIR / "train.jsonl"

MODEL_DIR = PROJECT_ROOT / "models"
QWEN_PATH = MODEL_DIR / "Qwen2.5-Math-1.5B"
SFT_PATH = MODEL_DIR / "Qwen2.5-Math-1.5B-sft"

OUTPUT_DIR = PROJECT_ROOT / "results"
PROMPT_PATH = CURRENT_DIR / "prompts" / "r1_zero.prompt"