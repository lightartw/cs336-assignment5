import argparse
from cs336_alignment.config import TrainingConfig
from cs336_alignment.sft import train


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-Math SFT Training Script")
    
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Total batch size")
    parser.add_argument("--max_iters", type=int, default=1000, help="Maximum number of iterations")
    parser.add_argument("--grad_accumulate_steps", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--device", type=str, default="cuda", help="Training device (default: cuda)")
    
    parser.add_argument("--num_examples", type=int, default=128, help="Number of dataset examples to use")
    parser.add_argument("--eval_interval", type=int, default=100, help="Steps between evaluations")
    parser.add_argument("--log_interval", type=int, default=10, help="Steps between logging")
    
    args = parser.parse_args()
    
    config = TrainingConfig(**vars(args))
    
    print("="*40)
    print("Training Configuration:")
    for key, value in config.model_dump().items():
        print(f"{key}: {value}")
    print("="*40)
    
    train(config)

if __name__ == "__main__":
    main()