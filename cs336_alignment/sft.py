import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from cs336_alignment.config import (
    QWEN_PATH, TRAIN_PATH, SFT_PATH
)

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








def main():
    model = AutoModelForCausalLM.from_pretrained(
        str(QWEN_PATH),
        torch_dtype=torch.bfloat16,
        atten_implementation="flash_attention_2",    
    )
    tokenizer = AutoTokenizer.from_pretrained(str(QWEN_PATH))
