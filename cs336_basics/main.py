import argparse
import os
import time

import numpy as np
import torch

from cs336_basics.loop import estimate_loss, get_batch, get_lr_cosine_schedule, gradient_clipping, load_checkpoint, save_checkpoint
from cs336_basics.module import Transformer_LM
from cs336_basics.training import AdamW, CrossEntropy, get_lr_cosine_schedule, gradient_clipping


"""
uv run python3 main.py --train_data /Users/YangWen/Documents/Code/github/assignment1-basics/data/TinyStoriesV2-GPT4-train-id.npy --val_data /Users/YangWen/Documents/Code/github/assignment1-basics/data/TinyStoriesV2-GPT4-valid-id.npy --vocab_size 10000
"""


def main(args):
    device = torch.device(args.device)

    # Memory-mapped datasets
    train_data = np.memmap(args.train_data, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.val_data, dtype=np.uint16, mode="r")

    model = Transformer_LM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
        device=device,
    )

    optimizer = AdamW(
        params=model.parameters(),
        lr=args.max_lr,
        weight_decay=args.weight_decay,
        betas=args.betas,
        eps=args.eps,
    )

    start_iter = 0
    if args.resume is not None:
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from iteration {start_iter}")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    t0 = time.time()

    for it in range(start_iter, args.max_iters):

        lr = get_lr_cosine_schedule(
            it=it,
            max_learning_rate=args.max_lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup,
            cosine_cycle_iters=args.cosine,
        )

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        x, y = get_batch(
            train_data,
            args.batch_size,
            args.context_length,
            device,
        )

        logits = model(x)
        loss = CrossEntropy(logits, y)

        loss.backward()

        gradient_clipping(model.parameters(), max_l2_norm=args.max_l2_norm)

        optimizer.step()

        # Logging
        if it % args.log_interval == 0 and it != 0:
            dt = time.time() - t0
            print(
                f"iter {it:6d} | "
                f"train loss {loss.item():.4f} | "
                f"time {dt:.2f}s"
            )
            t0 = time.time()

        # Validation
        if it % args.eval_interval == 0 and it > 0:
            model.eval()
            val_loss = estimate_loss(
                model,
                val_data,
                args.batch_size,
                args.context_length,
                device,
                args.eval_iters,
            )
            model.train()
            print(f"iter {it:6d} | val loss {val_loss:.4f}")

        # Checkpointing
        if it % args.ckpt_interval == 0 and it > 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_{it}.pt")
            save_checkpoint(model, optimizer, it, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    print("Training complete.")


# ------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)

    # Model
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--theta", type=float, default=10000)

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-6)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--cosine", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.995))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max_iters", type=int, default=100)
    parser.add_argument("--max_l2_norm", type=float, default=10)

    # Logging / eval
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--eval_iters", type=int, default=2)

    # Checkpoints
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--ckpt_interval", type=int, default=50)
    parser.add_argument("--resume", type=str, default=None)

    # Device
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    main(args)
