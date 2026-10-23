"""Full-data, paired PEMS08 retraining; run only via the project SSH/tslib GPU."""
import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from data_provider.data_factory import data_provider
from exp.exp_main import Exp_Main
from models.TimeXerGTR import Model
from scripts.adafe_diagnostics.common import (
    config, evaluate, forward_batch, initial_hash, runtime, save_json, seed_all,
)
from scripts.adafe_fixes.compare import VARIANTS
from scripts.adafe_fixes.result_index import append_completed
from utils.adafe_reporting import diagnostic_snapshot
from utils.tools import adjust_learning_rate

SOURCE_FILES = ['layers/adafe_v1.py', 'models/TimeXerGTR.py', 'models/TimeXer.py',
    'models/GTR.py', 'data_provider/data_factory.py', 'data_provider/data_loader.py',
    'exp/exp_main.py', 'utils/tools.py', 'utils/adafe_reporting.py', 'run.py',
    'scripts/adafe_diagnostics/common.py', 'scripts/adafe_fixes/compare.py']


def hashes():
    return {relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in SOURCE_FILES}


def make(job, device):
    args = config('PEMS08', job['seed'], batch=job['batch_size'], width=512,
                  epochs=job['epochs'], patience=job['patience'])
    vars(args).update(VARIANTS[job['variant']])
    args.pred_len = job['pred_len']
    args.factor = 3
    seed_all(args.random_seed)
    model = Model(args).to(device)
    experiment = Exp_Main.__new__(Exp_Main)
    experiment.args, experiment.model, experiment.device = args, model, device
    data, loaders = {}, {}
    for split in ('train', 'val', 'test'):
        dataset, loader = data_provider(args, split)
        # Separate sampler RNG, identical across all variants of a matched seed.
        loader.generator = torch.Generator().manual_seed(args.random_seed + 173)
        if split == 'train':
            loader.sampler.generator = loader.generator
        data[split], loaders[split] = dataset, loader
    return args, model, experiment, data, loaders


def capture(model, enabled):
    if model.use_adafe:
        model.adafe_adapter.adafe.capture_diagnostics = enabled


def gradient_check(model):
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError('Non-finite gradient in ' + name)
    if model.use_adafe:
        for name, parameter in model.adafe_adapter.adafe.score.named_parameters():
            if parameter.grad is None:
                raise AssertionError('Disconnected STE scorer ' + name)
        scorer = model.adafe_adapter.adafe.score
        if scorer.constant_reparam:
            assert not scorer.cos_coeffs.grad[:, :, 0].count_nonzero()
            assert not scorer.poly_coeffs.grad[:, :, 0].count_nonzero()


def probe(output, device):
    records = []
    for variant in ('baseline', 'legacy_ste', 'fixed_ste', 'fixed_slow'):
        job = dict(seed=2026, variant=variant, batch_size=32, epochs=30, patience=5, pred_len=24)
        args, model, experiment, data, loaders = make(job, device)
        optimizer = experiment._select_optimizer()
        torch.cuda.reset_peak_memory_stats()
        capture(model, True)
        started = time.monotonic()
        steps = []
        for index, batch in enumerate(loaders['train']):
            optimizer.zero_grad(set_to_none=True)
            prediction, target = forward_batch(model, batch, device)
            loss = F.mse_loss(prediction, target)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite probe loss')
            loss.backward()
            gradient_check(model)
            optimizer.step()
            steps.append(float(loss))
            if index == 9:
                break
        torch.cuda.synchronize()
        row = dict(variant=variant, steps=len(steps), seconds=time.monotonic() - started,
                   losses=steps, windows={key: len(value) for key, value in data.items()},
                   train_steps=len(loaders['train']), peak_cuda_mb=torch.cuda.max_memory_allocated() / 2**20,
                   diagnostics=diagnostic_snapshot(model))
        records.append(row)
        save_json(output / 'probe.json', records)
        print('PROBE', json.dumps(row), flush=True)
        del model, experiment, optimizer, data, loaders, prediction, target, loss
        gc.collect()
        torch.cuda.empty_cache()
    return records


def train_case(job, output, device, source_hashes):
    case = f"PEMS08_96_{job['pred_len']}_seed{job['seed']}_{job['variant']}"
    directory = output / 'runs' / case
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'result.json').exists():
        previous = json.loads((directory / 'configuration.json').read_text())
        if previous['job'] != job or previous['source_hashes'] != source_hashes:
            raise RuntimeError('Existing result has a different configuration')
        print('REUSED_COMPLETED', case, flush=True)
        return
    if hashes() != source_hashes:
        raise RuntimeError('Production source changed during this comparison')
    started = time.time()
    status = dict(state='building', case=case, pid=os.getpid(), started_unix=started)
    save_json(directory / 'status.json', status)
    args, model, experiment, data, loaders = make(job, device)
    shared_hash = initial_hash(model)
    scaler_hash = hashlib.sha256(data['train'].scaler.mean_.tobytes()
                                  + data['train'].scaler.scale_.tobytes()).hexdigest()
    windows = {split: len(dataset) for split, dataset in data.items()}
    index_hashes = {split: hashlib.sha256(np.arange(length, dtype=np.int64).tobytes()).hexdigest()
                    for split, length in windows.items()}
    save_json(directory / 'configuration.json', dict(args=vars(args), job=job,
        initial_shared_hash=shared_hash, initial_adafe_hash=initial_hash(model, True),
        scaler_hash=scaler_hash, windows=windows, index_hashes=index_hashes,
        source_hashes=source_hashes, physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        validation_shuffle=False, validation_drop_last=False, test_used_for_selection=False,
        train_windows_per_epoch=len(loaders['train']) * args.batch_size,
        sampler_seed=args.random_seed + 173, full_dataset_training=True,
        metric_aggregation='target_element_weighted_mean', metric_space='standardized'))
    optimizer = experiment._select_optimizer()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
        max_lr=[args.learning_rate * group['lr_scale'] for group in optimizer.param_groups],
        epochs=args.train_epochs, steps_per_epoch=len(loaders['train']), pct_start=args.pct_start)
    best, best_epoch, stale, history = float('inf'), 0, 0, []
    torch.cuda.reset_peak_memory_stats()
    status['state'] = 'training'
    try:
        for epoch in range(1, args.train_epochs + 1):
            epoch_started = time.time()
            model.train()
            total, count, last_gradient = 0., 0, {}
            for step, batch in enumerate(loaders['train'], 1):
                sampled = step == 1 or step % 100 == 0 or step == len(loaders['train'])
                capture(model, sampled)
                optimizer.zero_grad(set_to_none=True)
                prediction, target = forward_batch(model, batch, device)
                loss = F.mse_loss(prediction, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite training loss')
                loss.backward()
                if sampled:
                    gradient_check(model)
                    last_gradient = diagnostic_snapshot(model)
                    if not last_gradient['gradients_finite']:
                        raise FloatingPointError('Non-finite diagnostic norm')
                optimizer.step()
                adjust_learning_rate(optimizer, scheduler, epoch, args, False)
                scheduler.step()
                total += float(loss) * target.numel()
                count += target.numel()
                if sampled:
                    status.update(epoch=epoch, step=step, steps_per_epoch=len(loaders['train']),
                                  heartbeat_unix=time.time(), elapsed_seconds=time.time() - started,
                                  best_epoch=best_epoch, best_validation_mse=best if math.isfinite(best) else None)
                    save_json(directory / 'status.json', status)
                if time.time() - started > job['max_seconds']:
                    raise TimeoutError('Per-case time limit exceeded')
            capture(model, False)
            validation = experiment.vali(data['val'], loaders['val'], torch.nn.MSELoss())
            capture(model, True)
            model.eval()
            with torch.inference_mode():
                forward_batch(model, next(iter(loaders['val'])), device)
            diagnostics = diagnostic_snapshot(model)
            row = dict(epoch=epoch, train_mse=total / count, validation_mse=validation,
                       diagnostics=diagnostics, last_sampled_train_gradient=last_gradient,
                       lrs=scheduler.get_last_lr(), epoch_seconds=time.time() - epoch_started,
                       elapsed_seconds=time.time() - started)
            history.append(row)
            if validation < best:
                best, best_epoch, stale = validation, epoch, 0
                torch.save(model.state_dict(), directory / 'best.pth')
            else:
                stale += 1
            save_json(directory / 'history.json', history)
            status.update(epoch=epoch, step=len(loaders['train']), best_epoch=best_epoch,
                          best_validation_mse=best, stale_epochs=stale,
                          heartbeat_unix=time.time(), elapsed_seconds=time.time() - started)
            save_json(directory / 'status.json', status)
            print(f'EPOCH {case} {epoch}/{args.train_epochs} train={total/count:.7f} '
                  f'val={validation:.7f} best={best:.7f} stale={stale} '
                  f'sat={diagnostics["proxy_saturated_fraction"]}', flush=True)
            if stale >= args.patience:
                break
        if hashes() != source_hashes:
            raise RuntimeError('Production source changed before final evaluation')
        model.load_state_dict(torch.load(directory / 'best.pth', map_location=device, weights_only=True))
        status.update(state='testing', best_epoch=best_epoch)
        save_json(directory / 'status.json', status)
        capture(model, False)
        test = evaluate(model, loaders['test'], device)
        capture(model, True)
        with torch.inference_mode():
            forward_batch(model, next(iter(loaders['val'])), device)
        best_diagnostics = diagnostic_snapshot(model)
        result = dict(case=case, variant=job['variant'], seed=job['seed'], pred_len=args.pred_len,
            best_epoch=best_epoch, epochs_completed=len(history), validation_mse=best,
            test=test, best_diagnostics=best_diagnostics, elapsed_seconds=time.time() - started,
            full_dataset_training=True, windows=windows, initial_shared_hash=shared_hash,
            peak_cuda_mb=torch.cuda.max_memory_allocated() / 2**20, state='completed',
            test_evaluations=1, source_hashes=source_hashes)
        save_json(directory / 'result.json', result)
        status.update(state='completed', elapsed_seconds=time.time() - started)
        save_json(directory / 'status.json', status)
        print('COMPLETED', case, json.dumps(test), flush=True)
    except Exception as error:
        status.update(state='failed', error=f'{type(error).__name__}: {error}')
        save_json(directory / 'status.json', status)
        raise
    finally:
        del model, optimizer, scheduler, experiment, data, loaders
        gc.collect()
        torch.cuda.empty_cache()


def collect(output, plan):
    results, configurations = [], []
    for job in plan['jobs']:
        case = f"PEMS08_96_{job['pred_len']}_seed{job['seed']}_{job['variant']}"
        directory = output / 'runs' / case
        results.append(json.loads((directory / 'result.json').read_text()))
        configurations.append(json.loads((directory / 'configuration.json').read_text()))
    groups = []
    changed = {'use_adafe', 'adafe_constant_reparam', 'adafe_proxy_normalize', 'adafe_scorer_lr_scale'}
    for seed in plan['seeds']:
        group = [row for row in configurations if row['job']['seed'] == seed]
        comparable = [{key: value for key, value in row['args'].items() if key not in changed} for row in group]
        assert all(row == comparable[0] for row in comparable)
        assert len({row['initial_shared_hash'] for row in group}) == 1
        assert len({row['scaler_hash'] for row in group}) == 1
        assert all(row['source_hashes'] == group[0]['source_hashes'] for row in group)
        assert all(row['index_hashes'] == group[0]['index_hashes'] for row in group)
        assert all(row['validation_shuffle'] is False and row['validation_drop_last'] is False for row in group)
        groups.append(dict(seed=seed, variants=len(group), verified=True))
    means = {}
    for variant in plan['variants']:
        rows = [row for row in results if row['variant'] == variant]
        mse = [row['test']['mse'] for row in rows]
        mae = [row['test']['mae'] for row in rows]
        means[variant] = dict(mse_mean=statistics.mean(mse), mse_sd=statistics.stdev(mse),
                              mae_mean=statistics.mean(mae), mae_sd=statistics.stdev(mae), seeds=len(rows))
    paired = []
    for seed in plan['seeds']:
        rows = {row['variant']: row for row in results if row['seed'] == seed}
        base = rows['baseline']['test']['mse']
        old = rows['legacy_ste']['test']['mse']
        for variant in ('fixed_ste', 'fixed_slow'):
            value = rows[variant]['test']['mse']
            paired.append(dict(seed=seed, variant=variant, mse=value,
                mse_change_vs_baseline_pct=(value / base - 1) * 100,
                mse_change_vs_legacy_pct=(value / old - 1) * 100))
    save_json(output / 'summary.json', dict(plan=plan, results=results, means=means,
        paired=paired, matched_groups=groups, scope='PEMS08 full data, 96->24, three paired seeds'))
    lines = ['# PEMS08 完整数据配对结果', '',
        '输入 96、预测 24；完整训练/验证/测试划分；最多 30 轮、patience=5；batch=32、d_model=512。', '',
        '| seed | TimeXerGTR | 原始 STE | 修复 STE 默认 | 修复 STE + 打分器 LR×0.1 |',
        '|---|---:|---:|---:|---:|']
    for seed in plan['seeds']:
        rows = {row['variant']: row for row in results if row['seed'] == seed}
        lines.append('| ' + str(seed) + ' | ' + ' | '.join(f'{rows[variant]["test"]["mse"]:.7f}' for variant in plan['variants']) + ' |')
    lines.extend(['| 均值 | ' + ' | '.join(f'{means[variant]["mse_mean"]:.7f}' for variant in plan['variants']) + ' |',
                  '| 标准差 | ' + ' | '.join(f'{means[variant]["mse_sd"]:.7f}' for variant in plan['variants']) + ' |', '',
                  '所有方法使用修复后的验证口径；只根据验证 MSE 选模，选定后仅测试一次。结果仅对应 24 步预测。', ''])
    (output / 'comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    print('SUMMARY', json.dumps(means), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['probe', 'train', 'collect'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--plan')
    parser.add_argument('--shard', type=int, default=0)
    options = parser.parse_args()
    output = Path(options.output)
    device = runtime(output / f'{options.stage}_runtime_gpu{options.shard}')
    if options.stage == 'probe':
        probe(output, device)
        return
    plan = json.loads(Path(options.plan).read_text())
    if options.stage == 'collect':
        collect(output, plan)
        return
    source_hashes = hashes()
    failures = []
    for job in plan['jobs']:
        if job['shard'] != options.shard:
            continue
        try:
            train_case(job, output, device, source_hashes)
            case = f"PEMS08_96_{job['pred_len']}_seed{job['seed']}_{job['variant']}"
            added = append_completed(output / 'runs' / case)
            print('RESULT_INDEX', case, 'added' if added else 'already_present', flush=True)
        except Exception as error:
            failures.append(dict(job=job, error=f'{type(error).__name__}: {error}'))
            print('FAILED', json.dumps(failures[-1]), flush=True)
            if 'source changed' in str(error):
                break
    save_json(output / f'worker{options.shard}_finished.json', dict(failures=failures, state='completed'))
    if failures:
        raise RuntimeError(f'{len(failures)} cases failed')


if __name__ == '__main__':
    main()
