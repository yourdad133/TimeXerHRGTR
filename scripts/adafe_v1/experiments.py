"""Prepare matched commands, run short GPU checks, and summarize AdaFE V1.

Execution is restricted to SSH on the configured project host with tslib.
Formal training is explicit (--formal); --smoke never writes final test scores.
"""

import argparse
import ast
import copy
import csv
import gc
import hashlib
import json
import os
import random
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from data_provider.data_factory import data_provider
from models.TimeXerGTR import Model
from utils.adafe_reporting import (
    MATCHED_FIELDS, baseline_config, config_hash, diagnostic_snapshot,
    git_commit, write_json,
)

REPORT = ROOT / 'reports/adafe_v1'
REFERENCE = ROOT / 'scripts/adafe_v1/reference'
PYTHON = '/home/ouyangguanghong/miniconda3/envs/tslib/bin/python'
VERSIONS = [('B0', 0, 'hard'), ('A1_hard', 1, 'hard'), ('A2_ste', 1, 'ste')]


def check_runtime():
    connection = os.environ.get('SSH_CONNECTION', '').split()
    if (socket.gethostname() != 'ps' or len(connection) != 4
            or connection[-2:] != ['10.21.25.109', '22']
            or str(ROOT) != '/home/ouyangguanghong/projects/TEST/GTR-main'
            or sys.executable != PYTHON):
        raise RuntimeError('Run through project SSH on 10.21.25.109 using tslib.')
    subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used',
                    '--format=csv,noheader'], check=True)
    if not torch.cuda.is_available():
        raise RuntimeError('Remote CUDA unavailable; no CPU/local fallback.')
    print('Confirmed', socket.gethostname(), ROOT, sys.executable,
          'CUDA devices:', torch.cuda.device_count(), flush=True)


def parser_from_run():
    tree = ast.parse((ROOT / 'run.py').read_text())
    statements = []
    for node in tree.body:
        if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name)
                and t.id == 'parser' for t in node.targets)):
            statements.append(node)
        elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
              and isinstance(node.value.func, ast.Attribute)
              and isinstance(node.value.func.value, ast.Name)
              and node.value.func.value.id == 'parser'
              and node.value.func.attr == 'add_argument'):
            statements.append(node)
    namespace = {'argparse': argparse}
    exec(compile(ast.Module(body=statements, type_ignores=[]), 'run.py', 'exec'), namespace)
    return namespace['parser']


def matched_configs():
    parser = parser_from_run()
    pems_log = ROOT / 'experiment_logs/TimeXerGTR_PEMS08/PEMS08_96_24_seed2026.log'
    text = pems_log.read_text()
    line = next(line for line in text.splitlines() if line.startswith('Namespace('))
    call = ast.parse(line, mode='eval').body
    values = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
    pems = parser.parse_args(['--is_training', '1', '--model_id', 'AdaFEV1_PEMS08',
                              '--model', 'TimeXerGTR', '--data', 'PEMS'])
    vars(pems).update(values)
    # This reference is an existing TimeXer recipe; a flat GTR B0 must be run.
    weather_script = ROOT / 'scripts/baseline/TimeXer_Weather_M.sh'
    first_command = weather_script.read_text().split('python -u run.py', 1)[1].split('\n\n', 1)[0]
    tokens = shlex.split(first_command.replace('\\\n', ' ').replace('$model_name', 'TimeXerGTR'))
    weather = parser.parse_args(tokens)
    weather.gtr_cycle_len = weather.cycle
    weather.gtr_gate_init = .1
    configs = {'PEMS08': pems, 'Weather': weather}
    references = {
        'PEMS08': {'status': 'verified_existing_TimeXerGTR_run',
                   'source': str(pems_log.relative_to(ROOT)),
                   'historical_mse': .09660354256629944,
                   'historical_mae': .19260314106941223},
        'Weather': {'status': 'new_matched_GTR_B0_required',
                    'source': str(weather_script.relative_to(ROOT)),
                    'historical_mse': None, 'historical_mae': None,
                    'note': 'Flat GTR is added to the existing TimeXer recipe; '
                            'no verified flat TimeXerGTR Weather result was found.'},
    }
    return parser, configs, references


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_source(relative):
    path = REFERENCE / 'snapshots' / relative
    if path.exists():
        return path.read_bytes()
    backup = 'before-obsolete-model-cleanup-20261012'
    return subprocess.check_output(
        ['git', 'show', f'{backup}:reports/adafe_v1/snapshots/{relative}'],
        cwd=ROOT)


def snapshot_hash(relative):
    return hashlib.sha256(snapshot_source(relative)).hexdigest()


def cli_tokens(parser, config):
    include = set(MATCHED_FIELDS) | {
        'is_training', 'model', 'model_id', 'des', 'checkpoints', 'gpu',
        'use_adafe', 'adafe_mode', 'adafe_fourier_degree', 'adafe_poly_degree',
        'adafe_kernel', 'adafe_tau', 'adafe_gamma_init', 'adafe_report_dir',
        'adafe_log_interval',
    }
    result = []
    for action in parser._actions:
        if action.dest not in include:
            continue
        value = getattr(config, action.dest)
        if value is None:
            continue
        flag = next(option for option in action.option_strings if option.startswith('--'))
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                result.append(flag)
        elif isinstance(action, argparse._StoreFalseAction):
            if not value:
                result.append(flag)
        else:
            result.extend([flag, str(value)])
    return result


def prepare(gpu):
    REPORT.mkdir(parents=True, exist_ok=True)
    parser, configs, references = matched_configs()
    baseline = {
        'git_commit': git_commit(),
        'upstream_commit': '457d4477c9258e4d7be6b498dd723dce821baa74',
        'upstream_source_sha256': file_hash(REFERENCE / 'upstream/FeTS.py'),
        'upstream_license': 'No LICENSE or license declaration found at pinned revision.',
        'forward_path': 'input normalization -> unchanged GTR -> EnEmbedding -> '
                        'optional AdaFE (patches only) -> unchanged Encoder -> head',
        'global_token_index': -1, 'features_entry': 'M and MS share forecast()',
        'original_sources': {}, 'datasets': {},
    }
    for path in (REFERENCE / 'snapshots').rglob('*.py'):
        baseline['original_sources'][str(path.relative_to(REFERENCE / 'snapshots'))] = file_hash(path)
    lines = ['#!/usr/bin/env bash', 'set -euo pipefail',
             'source /home/ouyangguanghong/projects/TEST/GTR-main/scripts/adafe_v1/_runtime.sh',
             'adafe_check_runtime',
             '"$adafe_python" scripts/adafe_v1/experiments.py --check-formal']
    commands = []
    for dataset, base in configs.items():
        baseline['datasets'][dataset] = {
            'matched_config': baseline_config(base),
            'baseline_config_hash': config_hash(baseline_config(base)),
            'reference': references[dataset],
            'dataset_sha256': file_hash(ROOT / base.root_path / base.data_path),
        }
        for version, enabled, mode in VERSIONS:
            cfg = copy.deepcopy(base)
            cfg.model = 'TimeXerGTR'
            cfg.model_id = 'AdaFEV1_{}_{}'.format(dataset, version)
            cfg.use_adafe, cfg.adafe_mode = enabled, mode
            cfg.adafe_report_dir = 'reports/adafe_v1/runs/{}_sl{}_pl{}_seed{}_{}'.format(
                dataset, cfg.seq_len, cfg.pred_len, cfg.random_seed, version)
            cfg.checkpoints = 'reports/adafe_v1/checkpoints'
            cfg.gpu = 0  # CUDA_VISIBLE_DEVICES selects the physical card.
            command = ['env', 'CUDA_VISIBLE_DEVICES={}'.format(gpu), PYTHON, '-u',
                       'run.py'] + cli_tokens(parser, cfg)
            run_dir = shlex.quote(cfg.adafe_report_dir)
            lines.extend([
                '# {} / {} / seed {}'.format(dataset, version, cfg.random_seed),
                '[[ ! -e {} ]] || {{ echo "Run directory exists; choose a new name." >&2; exit 1; }}'.format(run_dir),
                'mkdir -p {}'.format(run_dir),
                'CUDA_VISIBLE_DEVICES={} adafe_check_runtime'.format(gpu),
                '{} 2>&1 | tee {}/train.log'.format(shlex.join(command), run_dir),
            ])
            commands.append({'dataset': dataset, 'version': version,
                             'baseline_config_hash': config_hash(baseline_config(base)),
                             'args': vars(cfg), 'command': command})
    (REPORT / 'stage_b_commands.sh').write_text('\n'.join(lines) + '\n')
    write_json(REPORT / 'baseline_config.json', baseline)
    write_json(REPORT / 'stage_b_commands.json', commands)
    print('Prepared six matched B0/A1/A2 commands; Weather B0 needs new training.', flush=True)
    return configs


def smoke(configs, datasets, gpu, batches, steps, run_id):
    device = torch.device('cuda:{}'.format(gpu))
    for dataset in datasets:
        base = configs[dataset]
        for batch_size in batches:
            for version, enabled, mode in VERSIONS:
                check_runtime()
                cfg = copy.deepcopy(base)
                cfg.use_adafe, cfg.adafe_mode = enabled, mode
                cfg.batch_size = batch_size
                cfg.num_workers = 0  # smoke only; official commands keep their original value
                cfg.use_multi_gpu = False
                directory = REPORT / 'smoke' / run_id / '{}_b{}_{}'.format(dataset, batch_size, version)
                directory.mkdir(parents=True, exist_ok=False)
                random.seed(cfg.random_seed)
                np.random.seed(cfg.random_seed)
                torch.manual_seed(cfg.random_seed)
                model = Model(cfg).float().to(device).train()
                train_data, loader = data_provider(cfg, 'train')
                if len(loader) < steps:
                    raise RuntimeError('Insufficient data for the smoke check.')
                optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
                torch.cuda.reset_peak_memory_stats(device)
                rows, elapsed = [], []
                iterator = iter(loader)
                for step in range(1, steps + 1):
                    batch = next(iterator)
                    x, y, xm, ym, cycle = [item.float().to(device) for item in batch[:5]]
                    cycle = cycle.long()
                    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                        raise FloatingPointError('Non-finite existing dataset; preprocessing is frozen.')
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.synchronize(device)
                    begin = time.perf_counter()
                    output = model(x, xm, None, ym, cycle)
                    target = y[:, -cfg.pred_len:, -1:] if cfg.features == 'MS' else y[:, -cfg.pred_len:]
                    loss = (output - target).square().mean()
                    if not torch.isfinite(output).all() or not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite AdaFE smoke output/loss.')
                    loss.backward()
                    row = {'step': step, 'train_mse': float(loss.detach())}
                    row.update(diagnostic_snapshot(model))
                    if not row['gradients_finite']:
                        raise FloatingPointError('Non-finite smoke gradients.')
                    for key in ('gtr_grad_norm', 'backbone_grad_norm'):
                        if not row[key] or row[key] <= 0:
                            raise RuntimeError('Detached or zero gradient: ' + key)
                    if enabled:
                        if not row['conv_grad_norm'] or row['conv_grad_norm'] <= 0:
                            raise RuntimeError('AdaFE convolution has no training gradient.')
                        if mode == 'hard' and row['scorer_missing_grad_count'] != row['scorer_parameter_count']:
                            raise RuntimeError('Hard scorer unexpectedly receives gradients.')
                        if mode == 'ste' and (row['scorer_missing_grad_count'] or not row['scorer_grad_norm']):
                            raise RuntimeError('STE scorer lacks training gradients.')
                    optimizer.step()
                    torch.cuda.synchronize(device)
                    elapsed.append(time.perf_counter() - begin)
                    rows.append(row)
                added = (sum(p.numel() for p in model.adafe_adapter.parameters()) if enabled else 0)
                metrics = {
                    'stage': 'smoke', 'dataset': dataset, 'version': version,
                    'seed': cfg.random_seed, 'seq_len': cfg.seq_len,
                    'pred_len': cfg.pred_len, 'cycle_len': cfg.gtr_cycle_len,
                    'batch_size': batch_size, 'steps': steps, 'physical_gpu': gpu,
                    'smoke_lr_schedule': 'constant', 'num_workers': 0,
                    'mse': None, 'mae': None, 'best_epoch': None,
                    'train_mse_first': rows[0]['train_mse'],
                    'train_mse_last': rows[-1]['train_mse'],
                    'param_total': sum(p.numel() for p in model.parameters()),
                    'param_added': added,
                    'peak_cuda_mb': torch.cuda.max_memory_allocated(device) / 1024**2,
                    'step_seconds': float(np.mean(elapsed[1:])),
                    'epoch_seconds': None,
                    'estimated_epoch_seconds': float(np.mean(elapsed[1:]) * len(loader)),
                    'mask_active_ratio': rows[-1]['mask_active_ratio'],
                    'gamma_final': float(model.adafe_adapter.gamma.detach()) if enabled else None,
                    'scorer_grad_norm': rows[-1]['scorer_grad_norm'],
                    'scorer_missing_grad_count': rows[-1]['scorer_missing_grad_count'],
                    'conv_grad_norm': rows[-1]['conv_grad_norm'],
                    'gtr_grad_norm': rows[-1]['gtr_grad_norm'],
                    'backbone_grad_norm': rows[-1]['backbone_grad_norm'],
                    'baseline_config_hash': config_hash(baseline_config(cfg)),
                    'baseline_reference_hash': config_hash(baseline_config(base)),
                    'git_commit': git_commit(),
                    'note': 'Short real-data training check; no final test scores or effectiveness claims.',
                }
                write_json(directory / 'config.json', vars(cfg))
                write_json(directory / 'metrics.json', metrics)
                with (directory / 'gate_diagnostics.csv').open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                print(json.dumps(metrics), flush=True)
                del model, optimizer, iterator, loader, train_data, x, y, xm, ym, cycle, output, target, loss
                gc.collect()
                torch.cuda.empty_cache()


def ensure_formal_ready(configs, datasets):
    for name in ('unit_tests.log', 'baseline_regression.log'):
        log = REPORT / name
        if not log.exists() or '\nOK\n' not in log.read_text():
            raise RuntimeError('Formal training blocked: {} has not passed. '
                               'Inspect the recorded failure; do not change '
                               'frozen preprocessing for the AdaFE comparison.'.format(name))
    for relative in ('data_provider/data_factory.py', 'data_provider/data_loader.py',
                     'models/GTR.py', 'models/TimeXer.py'):
        if file_hash(ROOT / relative) != snapshot_hash(relative):
            raise RuntimeError('Frozen baseline source changed: ' + relative)
    smoke_rows = [json.loads(path.read_text()) for path in (REPORT / 'smoke').glob('*/*/metrics.json')]
    for dataset in datasets:
        expected_hash = config_hash(baseline_config(configs[dataset]))
        for version, _, _ in VERSIONS:
            if not any(row['dataset'] == dataset and row['version'] == version
                       and row['baseline_reference_hash'] == expected_hash
                       and row['steps'] >= 20 for row in smoke_rows):
                raise RuntimeError('Missing matched 20-step smoke check for {} {}'.format(dataset, version))


def summarize():
    paths = sorted((REPORT / 'runs').glob('*/metrics.json'))
    paths += sorted((REPORT / 'smoke').glob('*/*/metrics.json'))
    rows = [json.loads(path.read_text()) for path in paths]
    fields = ['stage', 'dataset', 'version', 'seed', 'seq_len', 'pred_len', 'cycle_len',
              'batch_size', 'physical_gpu', 'smoke_lr_schedule', 'mse', 'mae',
              'param_total', 'param_added', 'peak_cuda_mb',
              'step_seconds', 'mask_active_ratio', 'gamma_final', 'scorer_grad_norm',
              'scorer_missing_grad_count', 'conv_grad_norm', 'gtr_grad_norm',
              'backbone_grad_norm', 'baseline_config_hash', 'git_commit']
    with (REPORT / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    formal = [row for row in rows if row['stage'] == 'formal']
    text = ['# AdaFE V1 results', '',
            'Hard uses the original FeTS mask. STE uses a surrogate backward gradient.',
            'Both use the same patch-only residual adapter, global threshold and hidden-feature convolution.', '',
            'Smoke rows are stability/performance checks; MSE/MAE are empty because no full test evaluation was run.',
            'Estimated epoch times in smoke metrics are extrapolations, not measured epochs.', '',
            'Formal comparisons use matching baseline configuration hashes, including dataset and seed.', '']
    if not formal:
        text.append('No formal B0/A1/A2 results yet. No accuracy improvement is claimed.')
    for variant in formal:
        if variant['version'] == 'B0':
            continue
        matched = next((base for base in formal if base['version'] == 'B0'
                        and base['baseline_config_hash'] == variant['baseline_config_hash']), None)
        if matched:
            mse_gain = 100 * (matched['mse'] - variant['mse']) / matched['mse']
            mae_gain = 100 * (matched['mae'] - variant['mae']) / matched['mae']
            text.append('{} {} seed {}: MSE improvement {:.3f}%, MAE improvement {:.3f}%.'.format(
                variant['dataset'], variant['version'], variant['seed'], mse_gain, mae_gain))
        else:
            text.append('{}: matched B0 is missing; no improvement is calculated.'.format(variant['version']))
    (REPORT / 'ablation.md').write_text('\n'.join(text) + '\n')
    # Aggregate per-step diagnostics with their run path for easy audit.
    diagnostics = []
    for metrics_path in paths:
        path = metrics_path.parent / 'gate_diagnostics.csv'
        if path.exists():
            with path.open() as handle:
                for row in csv.DictReader(handle):
                    diagnostics.append({'run': str(metrics_path.parent.relative_to(REPORT)), **row})
    if diagnostics:
        keys = list(dict.fromkeys(key for row in diagnostics for key in row))
        with (REPORT / 'gate_diagnostics.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(diagnostics)
    print('Summarized {} smoke/formal runs.'.format(len(rows)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--formal', action='store_true')
    parser.add_argument('--check-formal', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--datasets', nargs='+', choices=['PEMS08', 'Weather'], default=['PEMS08', 'Weather'])
    parser.add_argument('--batches', nargs='+', type=int, default=[2, 4])
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--run-id', default=time.strftime('%Y%m%d_%H%M%S'))
    options = parser.parse_args()
    check_runtime()
    if options.gpu < 0 or options.gpu >= torch.cuda.device_count():
        parser.error('Requested GPU is not visible.')
    if options.steps < 2 or any(batch < 1 for batch in options.batches):
        parser.error('Smoke requires steps >= 2 and positive batches.')
    configs = prepare(options.gpu) if options.prepare or options.formal else matched_configs()[1]
    if options.smoke:
        smoke(configs, options.datasets, options.gpu, options.batches, options.steps, options.run_id)
    if options.check_formal or options.formal:
        ensure_formal_ready(configs, options.datasets)
    if options.formal:
        commands = json.loads((REPORT / 'stage_b_commands.json').read_text())
        selected = [item for item in commands if item['dataset'] in options.datasets]
        if any((ROOT / item['args']['adafe_report_dir']).exists() for item in selected):
            raise FileExistsError('A selected formal run directory exists; choose new run names.')
        for item in selected:
            check_runtime()
            directory = ROOT / item['args']['adafe_report_dir']
            directory.mkdir(parents=True)
            print('Formal training:', item['dataset'], item['version'], flush=True)
            with (directory / 'train.log').open('w') as log:
                subprocess.run(item['command'], cwd=ROOT, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
    if options.summarize or options.smoke or options.formal:
        summarize()
