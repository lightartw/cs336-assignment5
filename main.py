import typer
from cs336_alignment.config import TrainingConfig
from cs336_alignment.sft import train as sft_train
from cs336_alignment.rl.config import GRPOConfig
from cs336_alignment.rl.grpo import train as grpo_train

app = typer.Typer(help="Qwen2.5-Math Alignment CLI")

@app.command()
def sft(
    lr: float = typer.Option(1e-3, help="Learning rate"),
    batch_size: int = typer.Option(32, help="Total batch size"),
    max_iters: int = typer.Option(50, help="Maximum number of iterations"),
    grad_accumulate_steps: int = typer.Option(16, help="Gradient accumulation steps"),
    device: str = typer.Option("cuda", help="Training device (default: cuda)"),
    num_examples: int = typer.Option(128, help="Number of dataset examples to use"),
    eval_interval: int = typer.Option(2, help="Steps between evaluations"),
    log_interval: int = typer.Option(1, help="Steps between logging"),
    dtype: str = typer.Option("bfloat16", help="dtype: bfloat16, float16, float32")
):
    """
    Qwen2.5-Math SFT Training Script
    """
    config = TrainingConfig(
        lr=lr,
        batch_size=batch_size,
        max_iters=max_iters,
        grad_accumulate_steps=grad_accumulate_steps,
        device=device,
        num_examples=num_examples,
        eval_interval=eval_interval,
        log_interval=log_interval,
        dtype=dtype
    )
    
    print("="*40)
    print("Training Configuration:")
    for key, value in config.model_dump().items():
        print(f"{key}: {value}")
    print("="*40)
    
    sft_train(config)

@app.command()
def grpo(
    n_steps: int = typer.Option(200, "--n-steps", help="Total GRPO rollout steps"),
    lr: float = typer.Option(1e-5, "--lr", help="Learning rate for policy model"),
    
    rollout_batch: int = typer.Option(256, "--rollout-batch", help="Total rollout samples per step"),
    group_size: int = typer.Option(8, "--group-size", help="Group size G for relative rewards"),
    train_batch: int = typer.Option(256, "--train-batch", help="Training batch size (should equal rollout_batch for on-policy)"),
    grad_accum: int = typer.Option(128, "--grad-accum", help="Gradient accumulation steps"),
    
    loss_type: str = typer.Option("reinforce_with_baseline", "--loss-type", help="Loss: no_baseline, reinforce_with_baseline, grpo_clip"),
    epochs_per_rollout: int = typer.Option(1, "--epochs", help="Training epochs per rollout batch"),
    use_std_norm: bool = typer.Option(True, "--std-norm/--no-std-norm", help="Whether to normalize advantages by standard deviation"),
    
    num_examples: int = typer.Option(512, "--num-examples", help="Total number of training questions to sample from"),
    eval_interval: int = typer.Option(20, "--eval-interval", help="Steps between evaluations"),
    log_interval: int = typer.Option(5, "--log-interval", help="Steps between logging"),
    dtype: str = typer.Option("bfloat16", help="dtype: bfloat16, float16, float32")
):
    """
    Qwen2.5-Math GRPO (Group Relative Policy Optimization) Training
    """
    config = GRPOConfig(
        n_grpo_steps=n_steps,
        learning_rate=lr,
        rollout_batch_size=rollout_batch,
        group_size=group_size,
        epochs_per_rollout_batch=epochs_per_rollout,
        train_batch_size=train_batch,
        gradient_accumulation_steps=grad_accum,
        loss_type=loss_type,
        use_std_normalization=use_std_norm,
        num_examples=num_examples,
        eval_interval=eval_interval,
        log_interval=log_interval,
        dtype=dtype
    )
    
    print("="*40)
    print("GRPO Training Configuration (All Adjustable Parameters):")
    for key, value in config.model_dump().items():
        print(f"{key}: {value}")
    print("="*40)
    
    grpo_train(config)

if __name__ == "__main__":
    # 运行 Typer 应用
    app()