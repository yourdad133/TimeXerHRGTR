"""Inspect existing Electricity weights using the repaired production module."""
import argparse
import copy
import gc
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from torch.utils.data._utils.collate import default_collate
from exp.exp_main import Exp_Main
from data_provider.data_factory import data_provider
from models.TimeXerGTR import Model
from scripts.adafe_diagnostics.common import (
    dataset_for, forward_batch, parser_from_run, runtime, save_json, seed_all,
)
from utils.adafe_reporting import diagnostic_snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--historical', type=Path,
                        default=ROOT / 'scripts/adafe_fixes/reference/historical_snapshot.json')
    options = parser.parse_args()
    output = Path(options.output)
    device = runtime(output / 'historical_probe_runtime')
    historical = json.loads(options.historical.read_text())
    rows = []
    for probe in historical['completed_ste_probes']:
        record = next(row for row in historical['records'] if row['file'] == probe['file'])
        args = parser_from_run().parse_args(['--is_training', '0', '--model_id', 'fixed_checkpoint_probe',
                                           '--model', 'TimeXerGTR', '--data', 'custom'])
        vars(args).update(record['args'])
        args.root_path, args.num_workers = str(ROOT / 'dataset'), 0
        checkpoint = ROOT / probe['checkpoint']
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        data = dataset_for(args, 'val')
        batch = default_collate([data[index] for index in probe['validation_indices']])
        reference_prediction = reference_mask = None
        row = dict(checkpoint=probe['checkpoint'], checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   prediction_length=args.pred_len, indices=probe['validation_indices'], variants={})
        for variant, fold, normalize in [('legacy_ste', 0, 0), ('proxy_only', 0, 1), ('fixed_ste', 1, 1)]:
            args.adafe_constant_reparam, args.adafe_proxy_normalize = fold, normalize
            seed_all(args.random_seed)
            model = Model(args).to(device).eval()
            model.load_state_dict(state, strict=True)
            model.adafe_adapter.adafe.capture_diagnostics = True
            captures = {}
            handle = model.adafe_adapter.adafe.score.register_forward_hook(
                lambda module, inputs, scores: captures.update(scores=scores.detach()))
            prediction, target = forward_batch(model, batch, device)
            handle.remove()
            loss = torch.nn.functional.mse_loss(prediction, target)
            loss.backward()
            scores = captures['scores']
            mask = scores > scores.mean()
            diagnostic = diagnostic_snapshot(model)
            if not diagnostic['gradients_finite'] or diagnostic['scorer_missing_grad_count']:
                raise AssertionError('Checkpoint probe lost finite scorer gradients')
            if reference_prediction is None:
                reference_prediction, reference_mask = prediction.detach().clone(), mask.clone()
            else:
                if variant == 'proxy_only':
                    torch.testing.assert_close(prediction, reference_prediction, atol=0, rtol=0)
                    assert torch.equal(mask, reference_mask)
            row['variants'][variant] = dict(mse=float(loss), **diagnostic,
                hard_mask_changed_fraction=float((mask != reference_mask).float().mean()),
                max_prediction_change=float((prediction.detach() - reference_prediction).abs().max()))
            if variant == 'fixed_ste' and args.pred_len == 96:
                args.batch_size = 32
                validation, loader = data_provider(args, 'val')
                experiment = Exp_Main.__new__(Exp_Main)
                experiment.args, experiment.model, experiment.device = args, model, device
                first = experiment.vali(validation, loader, torch.nn.MSELoss())
                second = experiment.vali(validation, loader, torch.nn.MSELoss())
                assert first == second, 'Repeated full validation changed'
                row['full_validation'] = dict(windows=len(validation), batches=len(loader),
                    batch_size=32, tail_windows=len(validation) % 32, drop_last=loader.drop_last,
                    sampler=type(loader.sampler).__name__, repeated_mse=[first, second],
                    absolute_repeat_difference=abs(first - second))
            del model, loss, prediction, target, captures
            gc.collect()
            torch.cuda.empty_cache()
        rows.append(row)
        save_json(output / 'historical_probes.json', rows)
        print('PROBED', args.pred_len, json.dumps(row['variants']), flush=True)
        del state, data, batch, reference_prediction, reference_mask
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
