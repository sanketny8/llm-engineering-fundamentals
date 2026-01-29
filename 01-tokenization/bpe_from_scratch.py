from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console

from llm_engineering_fundamentals.tokenization.bpe import BPEModel, demo_onehot_vs_learned

console = Console()


def _default_tiny_corpus() -> list[str]:
    return [
        "hello tokenization world",
        "tokenization is weird",
        "byte pair encoding learns merges",
        "hello hello hello",
        "world world world",
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--encode", action="store_true")
    p.add_argument("--decode", action="store_true")
    p.add_argument("--merges", type=int, default=200)
    p.add_argument("--out", type=str, default="01-tokenization/benchmarks/tiny_bpe.json")
    p.add_argument("--model", type=str, default="")
    p.add_argument("--text", type=str, default="hello tokenization")
    p.add_argument("--ids", type=str, default="")
    args = p.parse_args()

    if args.train:
        model = BPEModel.train(_default_tiny_corpus(), merges=args.merges)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(model.to_json(), indent=2))
        console.print(f"[green]Saved model[/green] -> {out}")
        stats = demo_onehot_vs_learned(vocab_size=64)
        console.print("[blue]Embedding demo stats:[/blue]")
        for k, v in stats.items():
            console.print(f"  {k}: {v:.4f}")
        return

    if args.encode:
        if not args.model:
            raise SystemExit("--encode requires --model")
        data = json.loads(Path(args.model).read_text())
        model = BPEModel.from_json(data)
        ids = model.encode(args.text)
        console.print({"text": args.text, "ids": ids})
        return

    if args.decode:
        if not args.model:
            raise SystemExit("--decode requires --model")
        if not args.ids:
            raise SystemExit("--decode requires --ids")
        data = json.loads(Path(args.model).read_text())
        model = BPEModel.from_json(data)
        ids = [int(x.strip()) for x in args.ids.split(",") if x.strip()]
        text = model.decode(ids)
        console.print({"ids": ids, "text": text})
        return

    raise SystemExit("Choose one: --train | --encode | --decode")


if __name__ == "__main__":
    main()


