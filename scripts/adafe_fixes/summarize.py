"""Collect completed verification evidence and assert paired-run comparability."""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.adafe_diagnostics.common import runtime, save_json

SOURCES = [
    'layers/adafe_v1.py', 'models/TimeXerGTR.py', 'data_provider/data_factory.py',
    'exp/exp_main.py', 'run.py', 'utils/tools.py', 'utils/adafe_reporting.py',
    'scripts/adafe_diagnostics/common.py',
    'scripts/adafe_diagnostics/test_diagnostics.py', 'tests/test_adafe_v1.py',
    'tests/test_adafe_fixes.py', 'scripts/adafe_fixes/compare.py',
    'scripts/adafe_fixes/plan.json', 'scripts/adafe_fixes/run_verify.sh',
    'scripts/adafe_fixes/probe_checkpoint.py',
    'scripts/adafe_fixes/check_preexisting_tests.py', 'scripts/adafe_fixes/summarize.py',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    options = parser.parse_args()
    output = Path(options.output)
    runtime(output / 'summary_runtime')
    plan = json.loads((ROOT / 'scripts/adafe_fixes/plan.json').read_text())
    for gpu in (0, 1):
        worker = json.loads((output / f'worker{gpu}_finished.json').read_text())
        assert not worker['failures'], worker
    contracts = (output / 'all_contracts.log').read_text()
    diagnostics = (output / 'diagnostic_contracts.log').read_text()
    assert re.search(r'Ran 21 tests.*?\nOK\b', contracts, re.S)
    assert re.search(r'Ran 9 tests.*?\nOK\b', diagnostics, re.S)
    audit = sum((json.loads((output / f'audit_gpu{gpu}.json').read_text()) for gpu in (0, 1)), [])
    assert len(audit) == 33 and sum(len(row['steps']) for row in audit) == 99
    assert all(step['gradients_finite'] for row in audit for step in row['steps'])
    results, configurations = [], []
    for job in plan['jobs']:
        case = f"{job['dataset']}_seed{job['seed']}_{job['variant']}"
        directory = output / 'runs' / case
        result = json.loads((directory / 'result.json').read_text())
        assert result['state'] == 'completed' and result['epochs_completed'] == job['epochs']
        results.append(result)
        configurations.append(json.loads((directory / 'configuration.json').read_text()))
    assert len(results) == 20
    changed = {'use_adafe', 'adafe_constant_reparam', 'adafe_proxy_normalize', 'adafe_scorer_lr_scale'}
    paired = []
    for dataset in ('Electricity', 'Weather', 'ETTm2', 'ETTh2'):
        rows = [cfg for cfg in configurations if cfg['job']['dataset'] == dataset]
        comparable_args = [{key: value for key, value in cfg['args'].items() if key not in changed} for cfg in rows]
        assert all(value == comparable_args[0] for value in comparable_args)
        assert len({cfg['initial_shared_hash'] for cfg in rows}) == 1
        assert all(cfg['index_hashes'] == rows[0]['index_hashes'] for cfg in rows)
        assert all(cfg['validation_shuffle'] is False and cfg['validation_drop_last'] is False for cfg in rows)
        paired.append(dict(dataset=dataset, variants=len(rows), shared_hash=rows[0]['initial_shared_hash'],
                           index_hashes=rows[0]['index_hashes'], windows=rows[0]['windows'], verified=True))
    preexisting = json.loads((output / 'preexisting_test_comparison.json').read_text())
    assert preexisting['identical_failure_sets']
    probes = json.loads((output / 'historical_probes.json').read_text())
    probe96 = next(row for row in probes if row['prediction_length'] == 96)
    assert probe96['full_validation']['absolute_repeat_difference'] == 0
    summary = dict(backup_commit=plan['backup_commit'], backup_tag='adafe-before-fixes-20261011',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        adafe_tests_passed=30, audit_combinations=33, real_updates=99, completed_training_cases=20,
        paired_groups=paired, historical_probes=probes, results=results,
        preexisting_gtr_tests=dict(tests_run=preexisting['current']['tests_run'],
            failures=len(preexisting['current']['failures']), errors=len(preexisting['current']['errors']),
            identical_to_prechange=True), full_dataset_training=False,
        scope='Three implementation defects repaired; short retraining is not a universal accuracy claim')
    save_json(output / 'summary.json', summary)
    manifest = []
    for relative in SOURCES:
        source = ROOT / relative
        target = output / 'code_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest.append(dict(file=relative, sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    save_json(output / 'code_snapshot/manifest.json', manifest)
    (output / 'training_comparison.md').write_text(
        '# 匹配短训结果\n\n'
        '全部为 seed 2026、6 轮、固定 512/257/257 个训练/验证/测试窗口；'
        'Electricity batch=8，其余 batch=16，宽度 512。所有变体采用完整尾批与加权验证。\n\n'
        '| 数据集 | TimeXerGTR | 原始 STE | 仅常数基修复 | 常数基+代理归一化 | 组合+打分器 LR×0.1 |\n'
        '|---|---:|---:|---:|---:|---:|\n' + '\n'.join(
            '| ' + dataset + ' | ' + ' | '.join(f'{next(row for row in results if row["dataset"] == dataset and row["variant"] == variant)["test"]["mse"]:.6f}'
                for variant in ('baseline', 'legacy_ste', 'constant_only', 'fixed_ste', 'fixed_slow')) + ' |'
            for dataset in ('Electricity', 'Weather', 'ETTm2', 'ETTh2')) +
        '\n\n这组测试 MSE 用于保留短预算正反例；不替代全量、多种子结果，不根据测试集切换默认配置。\n',
        encoding='utf-8')
    print(json.dumps(dict(adafe_tests=30, audit_combinations=33, updates=99, completed_cases=20,
                         matched_groups=len(paired), preexisting_failures_unchanged=True)))


if __name__ == '__main__':
    main()
