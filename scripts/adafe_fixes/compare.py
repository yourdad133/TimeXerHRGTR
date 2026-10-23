"""Remote-only audit and matched short retraining of production AdaFE fixes."""
import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F

from exp.exp_main import Exp_Main
from models.TimeXerGTR import Model
from scripts.adafe_diagnostics.common import (
    RECIPES, config, dataset_for, evaluate, forward_batch, initial_hash,
    loader_for, runtime, save_json, seed_all, window_indices,
)
from utils.adafe_reporting import diagnostic_snapshot
from utils.tools import adjust_learning_rate

VARIANTS = {
    'baseline': dict(use_adafe=0, adafe_constant_reparam=0, adafe_proxy_normalize=0, adafe_scorer_lr_scale=1.),
    'legacy_ste': dict(use_adafe=1, adafe_constant_reparam=0, adafe_proxy_normalize=0, adafe_scorer_lr_scale=1.),
    'constant_only': dict(use_adafe=1, adafe_constant_reparam=1, adafe_proxy_normalize=0, adafe_scorer_lr_scale=1.),
    'fixed_ste': dict(use_adafe=1, adafe_constant_reparam=1, adafe_proxy_normalize=1, adafe_scorer_lr_scale=1.),
    'fixed_slow': dict(use_adafe=1, adafe_constant_reparam=1, adafe_proxy_normalize=1, adafe_scorer_lr_scale=.1),
}


def build(dataset, variant, seed, batch, epochs, device):
    args = config(dataset, seed, batch=batch, width=512, epochs=epochs, patience=epochs)
    vars(args).update(VARIANTS[variant])
    seed_all(seed)
    model = Model(args).to(device)
    if model.use_adafe:
        model.adafe_adapter.adafe.capture_diagnostics = True
    experiment = Exp_Main.__new__(Exp_Main)
    experiment.args, experiment.model, experiment.device = args, model, device
    return args, model, experiment


def audit(output, device, shard, shards):
    rows = []
    for index, dataset in enumerate(RECIPES):
        if index % shards != shard:
            continue
        args = config(dataset, 2026, batch=2)
        data = dataset_for(args, 'train')
        hashes = []
        for variant, mode in [('legacy_ste', 'ste'), ('fixed_ste', 'ste'), ('fixed_ste', 'hard')]:
            args, model, experiment = build(dataset, variant, 2026, 2, 2, device)
            model.adafe_adapter.adafe.mode = mode
            hashes.append(initial_hash(model))
            optimizer = experiment._select_optimizer()
            loader = loader_for(data, 2, window_indices(len(data), 8), True, 2026)
            steps = []
            for step, batch in enumerate(loader):
                optimizer.zero_grad(set_to_none=True)
                prediction, target = forward_batch(model, batch, device)
                loss = F.mse_loss(prediction, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'{dataset}/{variant}/{mode} loss is non-finite')
                loss.backward()
                diagnostic = diagnostic_snapshot(model)
                if not diagnostic['gradients_finite']:
                    raise FloatingPointError(f'{dataset}/{variant}/{mode} gradient is non-finite')
                if mode == 'ste' and diagnostic['scorer_missing_grad_count']:
                    raise AssertionError('STE scorer is disconnected')
                if mode == 'hard' and diagnostic['scorer_missing_grad_count'] != 6:
                    raise AssertionError('Hard gradient semantics changed')
                optimizer.step()
                if variant == 'fixed_ste':
                    scorer = model.adafe_adapter.adafe.score
                    assert not scorer.cos_coeffs[:, :, 0].count_nonzero()
                    assert not scorer.poly_coeffs[:, :, 0].count_nonzero()
                steps.append(dict(step=step + 1, loss=float(loss), **diagnostic))
                if step == 2:
                    break
            rows.append(dict(dataset=dataset, variant=variant, mode=mode,
                             shared_hash=hashes[-1], steps=steps))
            save_json(output / f'audit_gpu{shard}.json', rows)
            del model, experiment, optimizer, loader
            gc.collect()
            torch.cuda.empty_cache()
        assert len(set(hashes)) == 1, 'Backbone/GTR initialization differs'
        print('AUDIT_COMPLETE', dataset, flush=True)
        del data
    return rows


def train_case(job, output, device):
    case = f"{job['dataset']}_seed{job['seed']}_{job['variant']}"
    directory = output / 'runs' / case
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'result.json').exists():
        raise FileExistsError(f'Completed case exists: {case}; use a new output directory')
    started = time.time()
    save_json(directory / 'status.json', dict(state='running', case=case, pid=os.getpid()))
    args, model, experiment = build(job['dataset'], job['variant'], job['seed'],
                                     job['batch_size'], job['epochs'], device)
    data = {split: dataset_for(args, split) for split in ('train', 'val', 'test')}
    indices = {split: window_indices(len(data[split]), job[split + '_windows']) for split in data}
    loaders = {split: loader_for(data[split], args.batch_size, indices[split],
                                split == 'train', args.random_seed) for split in data}
    shared_hash = initial_hash(model)
    save_json(directory / 'configuration.json', dict(args=vars(args), job=job,
        windows={split: len(idx) for split, idx in indices.items()},
        index_hashes={split: hashlib.sha256(idx.tobytes()).hexdigest() for split, idx in indices.items()},
        initial_shared_hash=shared_hash, validation_shuffle=False, validation_drop_last=False,
        validation_aggregation='target_element_weighted_mean', test_used_for_selection=False,
        physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
    optimizer = experiment._select_optimizer()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
        max_lr=[args.learning_rate * group['lr_scale'] for group in optimizer.param_groups],
        epochs=args.train_epochs, steps_per_epoch=len(loaders['train']), pct_start=args.pct_start)
    best, history, best_epoch = float('inf'), [], 0
    for epoch in range(1, args.train_epochs + 1):
        model.train()
        total, values = 0., 0
        for batch in loaders['train']:
            optimizer.zero_grad(set_to_none=True)
            prediction, target = forward_batch(model, batch, device)
            loss = F.mse_loss(prediction, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{case}: non-finite training loss')
            loss.backward()
            diagnostic = diagnostic_snapshot(model)
            if not diagnostic['gradients_finite']:
                raise FloatingPointError(f'{case}: non-finite gradient')
            optimizer.step()
            adjust_learning_rate(optimizer, scheduler, epoch, args, False)
            scheduler.step()
            total += float(loss) * target.numel()
            values += target.numel()
            if time.time() - started > job['max_seconds']:
                raise TimeoutError(f'{case}: time budget exceeded')
        # Use the production validator, including its tail-batch weighting.
        validation = experiment.vali(data['val'], loaders['val'], torch.nn.MSELoss())
        model.eval()
        with torch.inference_mode():
            forward_batch(model, next(iter(loaders['val'])), device)
        diagnostic = diagnostic_snapshot(model)
        row = dict(epoch=epoch, train_mse=total / values, validation_mse=validation,
                   diagnostics=diagnostic, lrs=scheduler.get_last_lr(), elapsed_seconds=time.time() - started)
        history.append(row)
        if validation < best:
            best, best_epoch = validation, epoch
            torch.save(model.state_dict(), directory / 'best.pth')
        save_json(directory / 'history.json', history)
        save_json(directory / 'status.json', dict(state='running', case=case, epoch=epoch, best_epoch=best_epoch))
        print(f'EPOCH {case} {epoch} val={validation:.7f} score_std={diagnostic["score_std"]} '
              f'sat={diagnostic["proxy_saturated_fraction"]}', flush=True)
    model.load_state_dict(torch.load(directory / 'best.pth', map_location=device, weights_only=True))
    test = evaluate(model, loaders['test'], device)
    with torch.inference_mode():
        forward_batch(model, next(iter(loaders['val'])), device)
    result = dict(case=case, dataset=job['dataset'], seed=job['seed'], variant=job['variant'],
        best_epoch=best_epoch, validation_mse=best, test=test, diagnostics=diagnostic_snapshot(model),
        epochs_completed=len(history), initial_shared_hash=shared_hash,
        elapsed_seconds=time.time() - started, state='completed', full_dataset_training=False)
    save_json(directory / 'result.json', result)
    save_json(directory / 'status.json', dict(state='completed', case=case))
    print('COMPLETED', case, json.dumps(test), flush=True)
    del model, optimizer, scheduler, experiment, loaders, data
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--shards', type=int, default=2)
    options = parser.parse_args()
    output = Path(options.output)
    device = runtime(output / f'worker{options.shard}_runtime')
    audit(output, device, options.shard, options.shards)
    jobs = json.loads(Path(options.plan).read_text())['jobs']
    failures = []
    for job in jobs:
        if job['shard'] != options.shard:
            continue
        try:
            train_case(job, output, device)
        except Exception as error:
            failures.append(dict(job=job, error=f'{type(error).__name__}: {error}'))
            case = f"{job['dataset']}_seed{job['seed']}_{job['variant']}"
            save_json(output / 'runs' / case / 'status.json', dict(state='failed', case=case, error=str(error)))
            print('FAILED', case, repr(error), flush=True)
    save_json(output / f'worker{options.shard}_finished.json', dict(failures=failures, state='completed'))
    if failures:
        raise RuntimeError(f'{len(failures)} cases failed')


if __name__ == '__main__':
    main()
