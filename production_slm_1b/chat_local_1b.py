"""
production_slm_1b/chat_local_1b.py

Interactive Local Terminal Chat with the 1B BitNet SLM:
Loads the 262MB packed 1.58-bit model directly into CPU memory.
Enables real-time question-answering, dialogue, and generation.
"""
import os
import sys
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../hopfield")))
from bitnet_1b_model import ProductionBitNet1B
from unpack_and_eval_local import unpack_ternary_tensor


def load_packed_1b_for_chat(ckpt_path: str, device="cpu"):
    print(f"--> Loading 1B SLM from: {ckpt_path} on {device}...")
    model = ProductionBitNet1B(dim=2048, n_layers=24, n_heads=16, hidden_dim=5632).to(device)
    raw = torch.load(ckpt_path, map_location=device)

    # Detect if packed or raw state_dict
    state_dict = {}
    if any(isinstance(v, dict) and v.get("is_ternary", False) for v in raw.values()):
        print("--> Unpacking 2-bit packed ternary representation...")
        for name, item in raw.items():
            if isinstance(item, dict) and item.get("is_ternary", False):
                w_q = unpack_ternary_tensor(item["packed"], item["shape"]).to(device)
                gamma = item["gamma"].to(device).float()
                state_dict[name] = w_q * gamma
            else:
                state_dict[name] = item.to(device).float()
    else:
        state_dict = raw.get("model_state_dict", raw)

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("--> 1B Model Ready for Chat!")
    return model


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7, top_k=40, device="cpu"):
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Truncate context to max length
        curr_input = input_ids[:, -1024:]
        logits = model(curr_input)[:, -1, :]

        # Top-k sampling
        if temperature > 0:
            logits = logits / temperature
            topk_vals, topk_idx = torch.topk(logits, k=top_k, dim=-1)
            probs = F.softmax(topk_vals, dim=-1)
            next_token_sub = torch.multinomial(probs, num_samples=1)
            next_token = topk_idx.gather(-1, next_token_sub)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if next_token.item() == tokenizer.eos_token_id or (tokenizer.eos_token and tokenizer.decode(next_token.item()) == "<|im_end|>"):
            break

    full_output = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)
    return full_output[len(prompt):].strip()


def chat_loop(model, tokenizer, device="cpu"):
    print("\n" + "=" * 65)
    print(" 1B BitNet SLM Interactive Terminal Chat (Type 'exit' to quit)")
    print("=" * 65)

    system_prompt = "<|im_start|>system\nYou are an intelligent, helpful 1B SLM assistant.<|im_end|>\n"
    history = system_prompt

    while True:
        try:
            user_input = input("\nUser > ").strip()
            if not user_input or user_input.lower() in ("exit", "quit"):
                break

            prompt = history + f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
            print("Assistant > ", end="", flush=True)

            reply = generate(model, tokenizer, prompt, max_new_tokens=120, temperature=0.7, device=device)
            print(reply)

            history += f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n{reply}<|im_end|>\n"
        except KeyboardInterrupt:
            print("\nExiting...")
            break


def main():
    parser = argparse.ArgumentParser(description="Chat with 1B BitNet SLM")
    parser.add_argument("--checkpoint", type=str, default="production_slm_1b/bitnet_1b_dialogue_chat.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = load_packed_1b_for_chat(args.checkpoint, device=device)
    chat_loop(model, tokenizer, device=device)


if __name__ == "__main__":
    main()
