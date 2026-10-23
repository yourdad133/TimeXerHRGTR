"""Plot direct Guide effects on identical patch tokens.

Run on the configured remote host:
CUDA_VISIBLE_DEVICES=0 /home/ouyangguanghong/miniconda3/envs/tslib/bin/python test.py

Edit PROBE_DIR, SAMPLE_ID and CHANNEL_ID below to choose the checkpoint/window.
The local mask removes only the direct scoring prior; main-route GTR stays on.
"""
import argparse
import csv
import io
import json
from datetime import datetime
from pathlib import Path
from types import MethodType

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from models.TimeXerGTRGuidedAdaFE import Model
from scripts.adafe_diagnostics.common import (
    dataset_for, forward_batch, runtime, seed_all,
)
from scripts.gtr_guided_adafe.probe_electricity96 import digest_bytes, digest_file

ROOT = Path(__file__).resolve().parent
PROBE_DIR = ROOT / "reports/electricity96_guide_probe_20261021_203122"
SAMPLE_ID = 0       # Window index in the validation dataset.
CHANNEL_ID = 0      # Variable index after Dataset_Custom reorders the target.
OUTPUT_DIR = ROOT / "reports" / (
    "mask_visualization_" + datetime.now().strftime("%Y%m%d_%H%M%S")
)


def plot_mask(mask, title, filename, context, changed=False):
    colors = ["#f8fafc", "#ea580c" if changed else "#2563eb"]
    fig, ax = plt.subplots(figsize=(14, 3.6))
    im = ax.imshow(
        mask.astype(np.uint8), aspect="auto", interpolation="nearest",
        cmap=ListedColormap(colors), vmin=0, vmax=1,
    )
    ax.set_xlabel("Hidden feature dimension")
    ax.set_ylabel("Patch index")
    ax.set_yticks(np.arange(mask.shape[0]))
    ax.set_title(title + "\n" + context)
    colorbar = fig.colorbar(im, ax=ax, ticks=[0, 1], pad=0.015)
    colorbar.ax.set_yticklabels(
        ["Unchanged", "Changed"] if changed else ["Dropped", "Selected"]
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close(fig)


def main():
    configuration = json.loads(
        (PROBE_DIR / "configuration.json").read_text(encoding="utf-8")
    )
    args = argparse.Namespace(**configuration["args"])
    if not args.guided_adafe_use_prior:
        raise ValueError("Select a checkpoint trained with direct Guide enabled.")
    args.root_path = str(ROOT / args.root_path)
    args.num_workers = 0
    checkpoint = ROOT / configuration["checkpoint"]
    checkpoint_bytes = checkpoint.read_bytes()
    checkpoint_sha = digest_bytes(checkpoint_bytes)
    if checkpoint_sha != configuration["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint changed since the recorded probe; check PROBE_DIR.")
    dataset_path = Path(args.root_path) / args.data_path
    if digest_file(dataset_path) != configuration["dataset_sha256"]:
        raise RuntimeError("Dataset changed since the recorded probe; check PROBE_DIR.")

    # Reuse the project's remote host / tslib / CUDA verification.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    device = runtime(OUTPUT_DIR)
    seed_all(args.random_seed)
    net = Model(args)
    state = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    net.load_state_dict(state, strict=True)
    del state, checkpoint_bytes
    net.to(device).eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)

    validation = dataset_for(args, "val")
    if not 0 <= SAMPLE_ID < len(validation):
        raise IndexError(f"SAMPLE_ID must be between 0 and {len(validation) - 1}.")
    if not 0 <= CHANNEL_ID < net.gtr_channels:
        raise IndexError(f"CHANNEL_ID must be between 0 and {net.gtr_channels - 1}.")
    batch = default_collate([validation[SAMPLE_ID]])
    batch_size = batch[0].shape[0]
    adafe = net.adafe_adapter.adafe
    original = adafe.conditioned_scores
    captured = {}

    def capture_scores(module, x, prior=None):
        # These are the scores actually returned to the native AdaFE forward.
        guided_scores = original(x, prior)
        if not captured:
            # Exactly the same x, normalization and learned local scorer.
            # The original x already contains the main-route GTR information.
            local_scores = module.score(module.score_norm(x))
            guided_mask = guided_scores > guided_scores.mean(dim=-1, keepdim=True)
            local_mask = local_scores > local_scores.mean(dim=-1, keepdim=True)
            flip = guided_mask != local_mask

            batch_channels = batch_size * net.gtr_channels
            if x.shape[0] % batch_channels:
                raise RuntimeError("Unexpected flattened patch-token layout.")
            patches = x.shape[0] // batch_channels
            shape = (batch_size, net.gtr_channels, patches, x.shape[-1])
            for name, value in [
                ("guided", guided_mask), ("local", local_mask), ("flip", flip),
                ("guided_scores", guided_scores), ("local_scores", local_scores),
            ]:
                captured[name] = (
                    value.reshape(shape)[0, CHANNEL_ID].detach().cpu().numpy().copy()
                )
            captured["token_layout"] = list(shape)
            captured["all_channels_flip_rate_this_window"] = float(flip.float().mean())
        return guided_scores

    adafe.conditioned_scores = MethodType(capture_scores, adafe)
    try:
        with torch.inference_mode():
            prediction, _ = forward_batch(net, batch, device)
    finally:
        adafe.conditioned_scores = original
    if not captured:
        raise RuntimeError("The model did not call the Guided AdaFE scorer.")
    with torch.inference_mode():
        reference, _ = forward_batch(net, batch, device)
    prediction_difference = float((prediction - reference).abs().max())
    if not torch.allclose(prediction, reference, atol=1e-7, rtol=1e-6):
        raise RuntimeError("Capturing masks unexpectedly changed the prediction.")

    with dataset_path.open(newline="", encoding="utf-8-sig") as stream:
        columns = next(csv.reader(stream))
    channel_names = [name for name in columns if name not in ("date", args.target)] + [args.target]
    phase = int(batch[4].reshape(-1)[0])
    context = (
        f"Validation window {SAMPLE_ID}; channel {CHANNEL_ID} "
        f"({channel_names[CHANNEL_ID]}); forecast phase {phase}/{net.gtr_cycle_len}"
    )
    plot_mask(captured["guided"], "GTR-guided feature mask", "guided_mask.png", context)
    plot_mask(captured["local"], "Local-score mask (direct guide removed)", "local_mask.png", context)
    plot_mask(captured["flip"], "Selection changes caused by direct periodic guidance",
              "mask_flip.png", context, changed=True)

    np.savez_compressed(
        OUTPUT_DIR / "masks.npz",
        **{name: captured[name] for name in
           ("guided", "local", "flip", "guided_scores", "local_scores")},
    )
    summary = {
        "split": "val", "sample_id": SAMPLE_ID, "channel_id": CHANNEL_ID,
        "channel_name": channel_names[CHANNEL_ID],
        "forecast_cycle_phase": phase, "gtr_cycle_len": net.gtr_cycle_len,
        "token_layout_B_N_P_D": captured["token_layout"],
        "plotted_shape_P_D": list(captured["guided"].shape),
        "guided_active_fraction": float(captured["guided"].mean()),
        "local_active_fraction": float(captured["local"].mean()),
        "selected_channel_flip_rate": float(captured["flip"].mean()),
        "all_channels_flip_rate_this_window": captured["all_channels_flip_rate_this_window"],
        "capture_max_abs_prediction_difference": prediction_difference,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_sha,
        "recorded_best_epoch": configuration["recorded_best_epoch"],
        "same_local_tokens": True, "main_route_gtr_enabled": True,
        "global_token_excluded": True,
        "threshold": "per-token mean over hidden features, strict greater-than",
        "scope": "Same trained model; local mask removes the direct scoring prior only.",
    }
    (OUTPUT_DIR / "mask_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("OUTPUT " + str(OUTPUT_DIR), flush=True)


if __name__ == "__main__":
    main()
