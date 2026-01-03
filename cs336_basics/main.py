import argparse
import os
import time

import numpy as np
import torch

from cs336_basics.data import estimate_loss, get_batch, load_checkpoint, plot_losses, save_checkpoint
from cs336_basics.decoder import decode
from cs336_basics.module import Transformer_LM
from cs336_basics.optimizer import AdamW, CrossEntropy, get_lr_cosine_schedule, gradient_clipping
from cs336_basics.tokenizer import Tokenizer


"""
uv run python3 main.py --train_data /workspace/yang.wen@xiaopeng.com/fm/xpilot_vision/cs336_basics/data/TinyStoriesV2-GPT4-train-id.npy --val_data /workspace/yang.wen@xiaopeng.com/fm/xpilot_vision/cs336_basics/data/TinyStoriesV2-GPT4-valid-id.npy --vocab_size 10000

uv run python3 main.py --train_data /Users/YangWen/Documents/Code/github/assignment1-basics/data/TinyStoriesV2-GPT4-train-id.npy --val_data /Users/YangWen/Documents/Code/github/assignment1-basics/data/TinyStoriesV2-GPT4-valid-id.npy --vocab_size 10000 --device mps --decoder --vocab_filepath /Users/YangWen/Documents/Code/github/assignment1-basics/data/TinyStoriesV2-GPT4-train_vocab.pkl --merges_filepath /Users/YangWen/Documents/Code/github/assignment1-basics/data/TinyStoriesV2-GPT4-train_merge.pkl --resume /Users/YangWen/Documents/Code/github/assignment1-basics/cs336_basics/checkpoints/ckpt_50.pt
"""

"""
7.2
learning rate
batch_size = 512
(a)
3e-4 final loss 1.4157 (10k steps) 1.5021 (5k steps)
5e-4 final loss 1.4229 (5k steps)
1e-3 final loss 1.3756 (5k steps)
5e-3 final loss 1.3697 (2.5k steps)
1e-1 final loss 3.22 (2.5k steps)
if it is less than 5e-3, it converges very slowly.
(b)
if it is large than 1e-1, it diverges
The edge of stability is the largest LR that does not diverge.
Divergence occurs when effective step size exceeds curvature limites.

batch_size experiment
small batch size -> noisy gradients
large batch size -> low noise gradients
linear scaling rule or square root scaling rule

generate
Today is a sunny day, we are going to the park today!" Sam agreed and they both smiled.
The sky was blue and the sun was shining. Tim and Sam watched the clouds as the sky got pink and orange. After playing, they looked up at the sky and said, "Wow, it's so pretty!"
Suddenly, a loud noise came from the sky. It was a plane falling down! Tim and Sam ran to see it. They watched as the plane hit the clouds. They felt dizzy.
Tim said, "I think the plane did fall!" Sam said, "Yes, it did! We got so dizzy!" They laughed and watched the plane fly away. They had a fun day at the park, even without the big, black plane.
<|endoftext|>

7.3
ablation 1
without RMSNorms, it divenges at the previous optimal learning rate.
Rmoving RMSNorm cuased training to diverge at the previous optimal learning rate. This indicates
that normalization plays a crucial role in stabilizing optimization by controlling activation and
gradient scales. Stability could be recovered by reducing the learning rate by an order of magnitude,
but training became significantly slower and converged to a worse solution. This demonstrates that
RMSNorm enables both higher learning rates and better optimization efficiency.
Lower LR does not fix scale drift, does not fix gradient anisotropy, makes optimization inefficient.
scale drift: magnitude of activations or gradients grows or shrinks as they pass through layers.
gradient anisotropy: gradient magnitudes vary wilding across different parameter directions, some
parameters very large gradients, some are very small gradients.

ablation 2
pre-norm vs post-norm
pre-norm keeps the residual path as an identity map while post-norm does not.
post-norm identity path is broken

ablation 3
SwiGLU vs SiLU
SwiGLU: what features to activation, how strongly to activate them,
gate can suppress noisy activations, reduces gradient explosion, improves conditioning
selective activation, implicit sparsity
SiLU can be useful if model is small

7.4
Using OWT data to traing the model with same setting (batch size = 256)
Loss is around 4.4127.
Loss is higher with OWT because OWT has higher intrinsic entropy.
owt: broad topics, long range dependencies, many valid next tokens
The model is worse than TinyStories based on loss.
This text is worse than TinyStories.
Fluency is OK. Fluency comes from local n-gram statistics, short-range syntactic patterns.
Cheap to learn. Missing long-range coherence, semantic depth, intentional structure
The token in OpenWebText is more than TinyStories.
According to scaling law, need to have large model and more compute budget.

Generate from owt
Today is a sunny day to Google to remove commercial satellites,” the term says.

While some of the most reliable phasers can download the picture but looks back at the moment.
Google suggested that it would be able to ship the most expensive software
using the tectonic crystals for lunar geometry.
Google started with an enhanced application,
set to expire on Wednesday and determine whether there would be somewhere between 3 and 13 in the past 10 years.
And yet, online file photo is now being told that using the researchers can probe
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

    model = torch.compile(model)
    start_iter = 0
    if args.resume is not None:
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from iteration {start_iter}")

    if args.decoder:
        EOS_TOKEN = "<|endoftext|>"
        mytokenizer = Tokenizer.from_files(
            vocab_filepath=args.vocab_filepath,
            merges_filepath=args.merges_filepath,
            special_tokens=[EOS_TOKEN],
        )
        prompt = mytokenizer.encode(args.prompt)
        inputs = torch.tensor([prompt], dtype=torch.int32, device=device)
        ans = decode(
            model=model,
            tokenizer=mytokenizer,
            prompt=inputs,
            max_token=args.max_tokens,
            temperature=args.temperature,
            threshold=args.threshold,
            eos_token=EOS_TOKEN,
            device=device,
        )
        print(ans)
        return

    os.makedirs(args.ckpt_dir, exist_ok=True)

    t0 = time.time()

    train_losses = []
    val_losses = []
    steps = []

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
        if it % args.log_interval == 0:
            dt = time.time() - t0
            print(
                f"iter {it:6d} | "
                f"train loss {loss.detach().item():.4f} | "
                f"time {dt:.2f}s"
            )
            steps.append(it)
            train_losses.append(loss.detach().item())
            t0 = time.time()

        # Validation
        if it % args.eval_interval == 0:
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
            val_losses.append(val_loss)
            print(f"iter {it:6d} | val loss {val_loss:.4f}")

        # Checkpointing
        if it % args.ckpt_interval == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_{it}.pt")
            save_checkpoint(model, optimizer, it, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    print("Training complete.")
    if len(steps) == len(train_losses) == len(val_losses) != 0:
        plot_losses(steps, train_losses, val_losses, args.max_lr, args.plot_dir)


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
    parser.add_argument("--max_iters", type=int, default=51)
    parser.add_argument("--max_l2_norm", type=float, default=10)

    # Logging / eval
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--eval_iters", type=int, default=2)

    # Checkpoints
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--ckpt_interval", type=int, default=50)
    parser.add_argument("--resume", type=str, default=None)

    # Plots
    parser.add_argument("--plot_dir", type=str, default="plots")

    # Device
    parser.add_argument("--device", type=str, default="cpu")

    # generate
    parser.add_argument("--decoder", action="store_true", help="Enable decoder mode (default: False)")
    parser.add_argument("--vocab_filepath", type=str, default=None)
    parser.add_argument("--merges_filepath", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--prompt", type=str, default="Today is a sunny day")


    args = parser.parse_args()
    main(args)
