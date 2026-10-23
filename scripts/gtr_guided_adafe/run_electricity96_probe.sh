#!/usr/bin/env bash
# Frozen validation diagnostics for the latest Electricity 96->96 Guide run.
set -euo pipefail
cd /home/ouyangguanghong/projects/TEST/GTR-main
export CUDA_VISIBLE_DEVICES=0
exec /home/ouyangguanghong/miniconda3/envs/tslib/bin/python -u scripts/gtr_guided_adafe/probe_electricity96.py
