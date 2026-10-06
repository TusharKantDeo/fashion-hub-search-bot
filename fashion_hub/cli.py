"""Interactive terminal chat.

    python -m fashion_hub.cli                    # metadata-filter strategy
    python -m fashion_hub.cli --strategy direct  # direct semantic search
"""

from __future__ import annotations

import argparse

from rag_kit.config import Settings

from fashion_hub.factory import build_bot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["metadata", "direct"], default="metadata")
    args = parser.parse_args()

    client, bot = build_bot(Settings.from_env(), strategy=args.strategy)
    print(f"Fashion Hub bot ({args.strategy}). Type 'exit' to quit, 'reset' to clear history.")
    try:
        while True:
            msg = input("\nYou: ").strip()
            if msg.lower() in {"exit", "quit"}:
                break
            if msg.lower() == "reset":
                bot.reset()
                continue
            if not msg:
                continue
            ans = bot.chat(msg)
            print(f"\nBot [{ans.route}]: {ans.content}")
            print(f"  ↳ {ans.ledger.summary()}  ~${ans.ledger.cost_usd():.6f}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
