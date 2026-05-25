"""Terminal progress for long hybrid jobs (tqdm if installed, else inline bar)."""

from __future__ import annotations

import sys
import time
from typing import Any, Callable, Iterable, Iterator, TypeVar

T = TypeVar("T")


def _use_tqdm() -> bool:
    try:
        import tqdm  # noqa: F401

        return True
    except ImportError:
        return False


class ShellProgress:
    """Simple \\r progress bar (no extra dependencies)."""

    def __init__(self, total: int, desc: str = "", *, width: int = 32) -> None:
        self.total = max(1, int(total))
        self.desc = desc[:48].ljust(48) if desc else ""
        self.width = width
        self.n = 0
        self.t0 = time.time()
        self._last_len = 0
        self._closed = False

    def update(self, n: int = 1, *, postfix: str = "") -> None:
        if self._closed:
            return
        self.n = min(self.total, self.n + int(n))
        pct = self.n / self.total
        filled = int(self.width * pct)
        bar = "=" * filled + "-" * (self.width - filled)
        elapsed = time.time() - self.t0
        eta = (elapsed / pct - elapsed) if pct > 0.02 else 0.0
        line = (
            f"\r{self.desc} [{bar}] {self.n:4d}/{self.total} "
            f"{pct * 100:5.1f}% {elapsed:6.0f}s eta {eta:5.0f}s"
        )
        if postfix:
            line += f"  {postfix[:40]}"
        pad = max(0, self._last_len - len(line))
        sys.stdout.write(line + " " * pad)
        sys.stdout.flush()
        self._last_len = len(line)

    def close(self, msg: str = "") -> None:
        if self._closed:
            return
        self._closed = True
        sys.stdout.write("\n")
        sys.stdout.flush()
        if msg:
            print(msg, flush=True)


def iter_progress(
    iterable: Iterable[T],
    *,
    desc: str = "",
    total: int | None = None,
    disable: bool = False,
) -> Iterator[T]:
    """Wrap *iterable* with tqdm or ShellProgress."""
    if disable:
        yield from iterable
        return

    if _use_tqdm():
        from tqdm import tqdm

        yield from tqdm(iterable, desc=desc, total=total)
        return

    items = list(iterable) if total is None else iterable
    n_total = total if total is not None else len(items)  # type: ignore[arg-type]
    bar = ShellProgress(n_total, desc=desc)
    try:
        for i, item in enumerate(items):  # type: ignore[arg-type]
            yield item
            bar.update(1)
    finally:
        bar.close()


def optuna_progress_callback(
    *,
    every: int = 10,
    disable: bool = False,
) -> Callable[[Any, Any], None]:
    """Optuna callback: print trial progress every *every* completed trials."""

    def _cb(study: Any, trial: Any) -> None:
        if disable:
            return
        n = len(study.trials)
        if n == 1 or n % every == 0 or trial.state.name == "COMPLETE" and n == study.trials[-1].number + 1:
            best = study.best_value if study.best_trial else float("nan")
            val = trial.value
            val_s = f"{val:.4f}" if val is not None and val > -1e5 else "pruned"
            feats = trial.user_attrs.get("features", [])
            nf = len(feats) if feats else "?"
            print(
                f"  [trial {n:4d}] value={val_s}  best={best:.4f}  n_feat={nf}",
                flush=True,
            )

    return _cb


def phase_banner(phase: str, index: int, total: int) -> None:
    print(f"\n{'=' * 70}", flush=True)
    print(f"PHASE {index}/{total}: {phase}", flush=True)
    print(f"{'=' * 70}", flush=True)
