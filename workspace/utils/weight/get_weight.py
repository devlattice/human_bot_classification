#!/usr/bin/env python3
"""
Fetch on-chain miner weights from a Bittensor metagraph.

W[i, j] = weight validator UID i assigns to UID j (see bittensor Metagraph.W).

Examples:
  python workspace/utils/weight/get_weight.py --uid 42
  python workspace/utils/weight/get_weight.py --hotkey 5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty
  python workspace/utils/weight/get_weight.py --block 12345678 --network archive
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Optional

import numpy as np

# Import bittensor only after CLI parse — a top-level import patches argparse globally.


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Read subnet weight matrix / weights toward one miner UID.")
    p.add_argument("--network", default="finney", help="Subtensor network (use 'archive' for old blocks).")
    p.add_argument("--netuid", type=int, default=126, help="Subnet netuid (Poker44 default: 126).")
    p.add_argument(
        "--block",
        type=int,
        default=None,
        help="Sync metagraph at this block (None = chain head). Old blocks may need --network archive.",
    )
    p.add_argument("--uid", type=int, default=None, help="Miner UID: print column W[:, uid].")
    p.add_argument(
        "--hotkey",
        default=None,
        help="Miner hotkey (ss58): resolve UID and print W[:, uid].",
    )
    p.add_argument(
        "--validators-only",
        action="store_true",
        help="Only print rows where validator_permit is true.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON (column or matrix summary).",
    )
    p.add_argument(
        "--top",
        type=int,
        default=20,
        help="Max nonzero validator rows (0 = all). Applies to text and --json.",
    )
    return p.parse_args()


def _resolve_uid(mg: Any, uid: Optional[int], hotkey: Optional[str]) -> Optional[int]:
    if uid is not None and hotkey is not None:
        raise SystemExit("Use only one of --uid or --hotkey.")
    if hotkey:
        try:
            return mg.hotkeys.index(hotkey)
        except ValueError:
            raise SystemExit(f"Hotkey not in metagraph for netuid: {hotkey!r}")
    return uid


def main() -> None:
    args = _parse_args()
    try:
        import bittensor as bt
    except ImportError as e:
        print("Install bittensor: pip install bittensor", file=sys.stderr)
        raise SystemExit(1) from e

    subtensor = bt.Subtensor(network=args.network)
    mg = bt.Metagraph(
        netuid=args.netuid,
        network=args.network,
        subtensor=subtensor,
        sync=False,
        lite=False,
    )
    mg.sync(block=args.block, lite=False, subtensor=subtensor)

    W = np.array(mg.W, dtype=np.float64)
    n = W.shape[0]
    if W.ndim != 2 or W.shape[1] != n:
        raise SystemExit(f"Unexpected W shape {W.shape}; expected (n, n).")

    target = _resolve_uid(mg, args.uid, args.hotkey)

    if args.json:
        out: dict[str, Any] = {
            "network": args.network,
            "netuid": args.netuid,
            "block": args.block,
            "synced_block": int(getattr(mg, "block", -1)) if getattr(mg, "block", None) is not None else None,
            "n": n,
            "w_shape": list(W.shape),
        }
        if target is not None:
            col = W[:, target]
            mask = np.array(mg.validator_permit, dtype=bool) if args.validators_only else np.ones(n, dtype=bool)
            pairs = [
                {"validator_uid": int(i), "weight": float(col[i]), "hotkey": mg.hotkeys[i]}
                for i in range(n)
                if mask[i] and col[i] > 0
            ]
            pairs.sort(key=lambda x: x["weight"], reverse=True)
            if args.top > 0:
                pairs = pairs[: args.top]
            out["miner_uid"] = target
            out["miner_hotkey"] = mg.hotkeys[target]
            out["weights_from_validators"] = pairs
            out["incentive_I"] = float(mg.I[target]) if hasattr(mg, "I") else None
        print(json.dumps(out, indent=2))
        return

    head = f"netuid={args.netuid} network={args.network}"
    if args.block is not None:
        head += f" block={args.block}"
    print(f"{head} | W shape {W.shape}")

    if target is None:
        print("No --uid/--hotkey: not printing a column. Use one of them, or --json for summary.")
        print(f"Nonzero entries in W: {int(np.count_nonzero(W))} / {W.size}")
        return

    col = W[:, target]
    print(f"Miner uid={target} hotkey={mg.hotkeys[target]}")
    if hasattr(mg, "I"):
        print(f"Incentive I[{target}] = {float(mg.I[target]):.8f}")

    mask = np.array(mg.validator_permit, dtype=bool) if args.validators_only else np.ones(n, dtype=bool)
    idx = np.where(mask & (col > 0))[0]
    sorted_idx = idx[np.argsort(-col[idx])]
    order = sorted_idx if args.top <= 0 else sorted_idx[: args.top]

    print(f"Top {len(order)} weights W[validator_uid, {target}] (validators_only={args.validators_only}):")
    for i in order:
        hk = mg.hotkeys[i]
        hk_short = hk[:12] + "…" + hk[-6:] if len(hk) > 20 else hk
        vp = bool(mg.validator_permit[i]) if hasattr(mg, "validator_permit") else False
        print(f"  uid={i:4d}  weight={col[i]:.8f}  validator_permit={vp}  hotkey={hk_short}")


if __name__ == "__main__":
    main()
