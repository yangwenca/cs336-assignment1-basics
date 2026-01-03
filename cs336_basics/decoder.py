from cs336_basics.module import Softmax, Transformer_LM
from cs336_basics.tokenizer import Tokenizer
import torch


def decode(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompt: torch.Tensor,
    max_token: int,
    temperature: float = 1.0,
    threshold: float = 0.9,
    eos_token: str = '<|endoftext|>',
    device: str = "cpu",
) -> str:
    model.eval()
    model.to(device)

    end_idx = tokenizer.encode(eos_token)[0]
    generated = prompt.clone()
    while generated.size(-1) < max_token:
        output = model(generated)
        # temperature scaling
        logits = output[:, -1, :] / max(temperature, 1e-8)

        # top-p sampling
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(Softmax(in_features=sorted_logits, dim=-1), dim=-1)
        # mask tokens above top-p threshold
        sorted_indices_to_remove = cumulative_probs > threshold
        # shift one right to keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        sorted_logits[sorted_indices_to_remove] = -float('inf')
        probs = Softmax(sorted_logits, dim=-1)

        # sample
        next_token = torch.multinomial(probs, num_samples=1)
        next_token_id = sorted_indices.gather(-1, next_token)
        # stop if EOS
        if end_idx == next_token_id.item():
            break
        generated = torch.cat([generated, next_token_id], dim=-1)

    # Decode generated tokens to string
    output_ids = generated[0].tolist()
    generated_text = tokenizer.decode(output_ids)
    return generated_text


def test_decode() -> None:
    prefix = "/Users/YangWen/Documents/Code/github/assignment1-basics/data/"
    vocab_size = 10000
    context_length = 128
    d_model = 256
    num_layers = 12
    num_heads = 8
    d_ff = 1024
    theta= 10000
    eos_token = "<|endoftext|>"

    myTokenzier = Tokenizer.from_files(
        vocab_filepath=prefix + "TinyStoriesV2-GPT4-train_vocab.pkl",
        merges_filepath=prefix + "TinyStoriesV2-GPT4-train_merge.pkl",
        special_tokens=[eos_token],
    )

    model = Transformer_LM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        theta=theta,
    )
    prompt = myTokenzier.encode("today is a sunny day")
    inputs = torch.tensor([prompt], dtype=torch.int32)
    ans = decode(
        model=model,
        tokenizer=myTokenzier,
        prompt=inputs,
        max_token=50,
        temperature=1.0,
        eos_token=eos_token,
    )
    print(ans)

# test_decode()
