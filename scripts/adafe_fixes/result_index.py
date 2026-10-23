"""Append completed full-data AdaFE experiments to the project's result.txt."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def append_completed(directory, target=None):
    directory = Path(directory).resolve()
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    configuration = json.loads((directory / "configuration.json").read_text(encoding="utf-8"))
    args = configuration["args"]
    if result["state"] != "completed" or not result["full_dataset_training"]:
        raise ValueError("Only completed full-data experiments belong in result.txt")
    if configuration["metric_space"] != "standardized":
        raise ValueError("Unexpected metric space")
    if result["seed"] != args["random_seed"] or result["pred_len"] != args["pred_len"]:
        raise ValueError("Result and configuration disagree")
    mse, mae = result["test"]["mse"], result["test"]["mae"]
    if not all(math.isfinite(value) for value in (mse, mae)):
        raise ValueError("Cannot index non-finite test metrics")
    relative = (directory / "result.json").relative_to(ROOT).as_posix()
    marker = f"result_source={relative}"
    setting = (
        f'{result["case"]}_TimeXerGTR_adafe{args["use_adafe"]}_{args["adafe_mode"]}'
        f'_cr{args["adafe_constant_reparam"]}_ps{args["adafe_proxy_normalize"]}'
        f'_slr{args["adafe_scorer_lr_scale"]:g}'
    )
    block = f"{setting} [metric_space=standardized; {marker}]\nmse:{mse}, mae:{mae}\n\n"
    target = ROOT / "result.txt" if target is None else Path(target)
    # Both GPU workers may finish together; preserve complete, unique records.
    with target.open("a+", encoding="utf-8", newline="\n") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        existing = handle.read()
        if marker in existing:
            return False
        handle.seek(0, os.SEEK_END)
        handle.write(("\n" if existing and not existing.endswith("\n") else "") + block)
        handle.flush()
        os.fsync(handle.fileno())
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", nargs="+", required=True)
    options = parser.parse_args()
    added = skipped = 0
    for output in options.output:
        for result_file in sorted((Path(output) / "runs").glob("*/result.json")):
            if append_completed(result_file.parent):
                added += 1
            else:
                skipped += 1
    print(json.dumps(dict(target=str(ROOT / "result.txt"), added=added, already_present=skipped)),
          flush=True)


if __name__ == "__main__":
    main()
