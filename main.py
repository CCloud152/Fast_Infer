"""CLI entry point for Fast_Infer interactive chat."""

import argparse
import sys
from pathlib import Path
from fast_infer.engine import InferenceEngine


def main():
    parser = argparse.ArgumentParser(description="Fast_Infer — Triton-based Llama 3.2 inference")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to downloaded model directory")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max new tokens per turn")
    parser.add_argument("--repetition-penalty", type=float, default=1.1,
                        help="Repetition penalty (1.0=none, 1.1=moderate)")
    args = parser.parse_args()

    model_path = Path(args.model_dir)
    if not model_path.exists():
        print(f"Error: model directory not found: {args.model_dir}")
        print("Download first: python -m fast_infer.download")
        sys.exit(1)

    print("Starting Fast_Infer...")
    engine = InferenceEngine(str(model_path))

    print(f"\nModel loaded. temp={args.temperature}, top_p={args.top_p}, "
          f"max_tokens={args.max_tokens}, rep_penalty={args.repetition_penalty}")
    print("Enter prompts below (Ctrl+C to exit).\n")

    try:
        while True:
            prompt = input("> ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("quit", "exit", "q"):
                break

            print()
            for token_id in engine.generate_stream(
                prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            ):
                text = engine.tokenizer.decode([token_id], skip_special_tokens=True)
                print(text, end="", flush=True)
            print("\n")

    except KeyboardInterrupt:
        print("\nGoodbye.")


if __name__ == "__main__":
    main()
