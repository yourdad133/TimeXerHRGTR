"""Matched guide on/off training with RevIN disabled.

This focused entry point reuses the verified pilot engine without changing
its frozen source or prior experiment hashes. All trained arms use RevIN=0.
Defaults remain 1,024 train windows, 256 validation windows and six epochs.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.gtr_guided_adafe import run_pilot as engine

BASE_MAKE_ARGS = engine.make_args
NORM0_ARMS = (
    ("guided_without_prior_norm0", True, True, 0),
    ("guided_with_prior_norm0", True, True, 0),
)


def norm0_args(options, arm, enabled, use_norm):
    args = BASE_MAKE_ARGS(options, arm, enabled, use_norm)
    args.guided_adafe_use_prior = arm == "guided_with_prior_norm0"
    if arm in {spec[0] for spec in NORM0_ARMS}:
        if args.use_revin != 0:
            raise AssertionError("Every trained norm0 arm must disable RevIN")
        if args.guided_adafe_use_prior != arm.startswith("guided_with_prior_"):
            raise AssertionError("Periodic-guide flag does not match the declared arm")
    return args


def main():
    engine.ARMS = NORM0_ARMS
    engine.make_args = norm0_args
    engine.SOURCE_FILES = engine.SOURCE_FILES + (
        "scripts/gtr_guided_adafe/run_norm0_pilot.py", "run.py",
    )
    engine.main()


if __name__ == "__main__":
    main()

