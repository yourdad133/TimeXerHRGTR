#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_runtime.sh"
adafe_check_runtime
mkdir -p reports/adafe_v1
"$adafe_python" scripts/adafe_v1/experiments.py --prepare
"$adafe_python" -m unittest discover -s tests -p test_adafe_v1.py -v 2>&1 | tee reports/adafe_v1/unit_tests.log
"$adafe_python" -m unittest discover -s tests -p 'test_timexer_gtr*.py' -v 2>&1 | tee reports/adafe_v1/baseline_regression.log
