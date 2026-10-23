"""Fixed-config, twelve-epoch Electricity diagnostic; production code is unchanged."""
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import exp.exp_main as exp_module
from scripts.baseline_gtr.run_gtr_guided_adafe import (
    GuidedExperiment, check_remote_runtime, experiment_setting, parse_args,
)
from scripts.adafe_diagnostics.common import seed_all
from utils.tools import EarlyStopping

SNAPSHOT_EPOCHS = (4, 8, 12)
STOP_EPOCH = 12


def timestamp():
    return datetime.now().astimezone().isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def gate_values(model):
    return {
        'gtr_gamma': float(model.gamma.detach().cpu()),
        'adafe_gamma': float(model.adafe_adapter.gamma.detach().cpu()),
        'guide_prior_gate_effective': float(model.adafe_adapter.adafe.prior_gate.detach().tanh().cpu()),
    }


def save_snapshot(path, model, args, epoch, losses, gates):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    payload = {'epoch': int(epoch), 'model_state_dict': state,
               'args': vars(args).copy(), 'losses': dict(losses), 'gates': dict(gates)}
    temporary = path.with_name(path.name + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)
    print(f'DIAGNOSTIC_SNAPSHOT epoch={epoch} path={path}', flush=True)


class DiagnosticEarlyStopping(EarlyStopping):
    """Keep validation-best selection; independently stop the diagnostic at 12."""
    native_first_stop_epoch = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epochs_seen = 0

    def __call__(self, val_loss, model, path):
        self.epochs_seen += 1
        super().__call__(val_loss, model, path)
        if self.early_stop and type(self).native_first_stop_epoch is None:
            type(self).native_first_stop_epoch = self.epochs_seen
            print(f'DIAGNOSTIC_NATIVE_EARLY_STOP epoch={self.epochs_seen}', flush=True)
        self.early_stop = self.epochs_seen >= STOP_EPOCH
        if self.early_stop:
            print(f'DIAGNOSTIC_STOP twelve-epoch cap; native_first_stop_epoch={type(self).native_first_stop_epoch}', flush=True)


class DiagnosticExperiment(GuidedExperiment):
    def __init__(self, args, run_dir):
        self.run_dir = Path(run_dir)
        self.epoch_gates = []
        self.last_recorded_epoch = 0
        self.loss_history = []
        super().__init__(args)

    def _write_training_history(self, history_path, loss_history):
        super()._write_training_history(history_path, loss_history)
        self.loss_history = list(loss_history)
        write_json(self.run_dir / 'training_history.json', loss_history)
        if not loss_history:
            return
        row = loss_history[-1]
        epoch = int(row['epoch'])
        # The production train() writes history again after loading the best model.
        if epoch <= self.last_recorded_epoch:
            return
        self.last_recorded_epoch = epoch
        gates = gate_values(self.model)
        self.epoch_gates.append({'epoch': epoch, **gates})
        write_json(self.run_dir / 'epoch_gates.json', self.epoch_gates)
        if epoch in SNAPSHOT_EPOCHS:
            save_snapshot(self.run_dir / 'snapshots' / f'epoch{epoch:02d}.pth',
                          self.model, self.args, epoch, row, gates)
        write_json(self.run_dir / 'status.json', {
            'state': 'training', 'completed_epochs': epoch,
            'updated_at': timestamp(), 'run_dir': str(self.run_dir),
        })


def preserve_reference(args, run_dir):
    original_args = copy.copy(args)
    original_args.model_id = 'Electricity_96_720Guide'
    original_setting = experiment_setting(original_args)
    files = {
        'source_log.log': ROOT / 'experiment_logs/Guide_Electricity/Electricity_720_2026Guide_dff1024.log',
        'training_history.json': ROOT / 'results' / original_setting / 'training_history.json',
        'checkpoint.pth': ROOT / 'checkpoints' / original_setting / 'checkpoint.pth',
    }
    reference = run_dir / 'reference'
    reference.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, source in files.items():
        destination = reference / name
        shutil.copy2(source, destination)
        manifest[name] = {'source': str(source), 'copy': str(destination),
                          'sha256': sha256(source), 'copy_sha256': sha256(destination),
                          'bytes': source.stat().st_size}
        if manifest[name]['sha256'] != manifest[name]['copy_sha256']:
            raise RuntimeError(f'Reference copy hash mismatch: {source}')
    write_json(reference / 'manifest.json', manifest)
    return manifest


def main():
    args = parse_args()
    check_remote_runtime(args)
    run_dir = Path(os.environ['DIAGNOSTIC_DIR']).resolve()
    run_dir.relative_to(ROOT / 'reports')
    run_dir.mkdir(parents=True, exist_ok=True)
    for artifact in ('config.json', 'status.json', 'snapshots'):
        if (run_dir / artifact).exists():
            raise FileExistsError(f'Diagnostic output already exists: {run_dir / artifact}')
    args.checkpoints = str(run_dir / 'checkpoints')
    args.model_id = run_dir.name
    if args.train_epochs != 30:
        raise ValueError('Keep train_epochs=30: it defines the original OneCycle schedule')
    print(f'DIAGNOSTIC_DIR={run_dir}', flush=True)
    print('Effective model: models.TimeXerGTRGuidedAdaFE.Model', flush=True)
    print(args, flush=True)
    source_paths = [
        'exp/exp_main.py', 'utils/tools.py', 'models/TimeXerGTRGuidedAdaFE.py',
        'models/TimeXerGTR.py', 'models/GTR.py', 'layers/gtr_guided_adafe.py',
        'models/TimeXer.py', 'scripts/baseline_gtr/run_gtr_guided_adafe.py',
        'scripts/electricity720_diagnostic/run_diagnostic.py',
    ]
    try:
        reference = preserve_reference(args, run_dir)
        write_json(run_dir / 'config.json', {
            'args': vars(args), 'created_at': timestamp(), 'run_dir': str(run_dir),
            'diagnostic_stop_epoch': STOP_EPOCH, 'snapshot_epochs': SNAPSHOT_EPOCHS,
            'schedule_train_epochs': args.train_epochs,
            'native_patience': args.patience,
            'stop_rule': 'Preserve native validation-best saves; suppress exit before epoch 12 and stop at 12.',
            'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES'),
            'source_sha256': {rel: sha256(ROOT / rel) for rel in source_paths},
            'dataset_sha256': sha256(ROOT / args.root_path / args.data_path),
            'reference_files': reference,
        })
        write_json(run_dir / 'status.json', {'state': 'initializing', 'updated_at': timestamp()})
        seed_all(args.random_seed)
        DiagnosticEarlyStopping.native_first_stop_epoch = None
        experiment = DiagnosticExperiment(args, run_dir)
        setting = experiment_setting(args)
        original_early_stopping = exp_module.EarlyStopping
        try:
            exp_module.EarlyStopping = DiagnosticEarlyStopping
            experiment.train(setting)
        finally:
            exp_module.EarlyStopping = original_early_stopping
        history = experiment.loss_history
        minimum = min(row['validation_mse_standardized'] for row in history)
        # Original EarlyStopping saves the latest model when the minimum ties.
        best_row = [row for row in history if row['validation_mse_standardized'] == minimum][-1]
        best_epoch = int(best_row['epoch'])
        best_gates = next(row for row in experiment.epoch_gates if row['epoch'] == best_epoch)
        best_path = run_dir / 'snapshots' / 'validation_best.pth'
        # train() has already restored the real validation-best model here.
        save_snapshot(best_path, experiment.model, args, best_epoch, best_row, best_gates)
        write_json(run_dir / 'status.json', {
            'state': 'evaluating', 'completed_epochs': len(history), 'best_epoch': best_epoch,
            'native_first_stop_epoch': DiagnosticEarlyStopping.native_first_stop_epoch,
            'updated_at': timestamp(),
        })
        from analysis_metrics import evaluate_checkpoints
        paths = [run_dir / 'snapshots' / f'epoch{epoch:02d}.pth' for epoch in SNAPSHOT_EPOCHS]
        paths.append(best_path)
        summary = evaluate_checkpoints(experiment, args, run_dir, paths)
        write_json(run_dir / 'status.json', {
            'state': 'complete', 'completed_epochs': len(history), 'best_epoch': best_epoch,
            'native_first_stop_epoch': DiagnosticEarlyStopping.native_first_stop_epoch,
            'finished_at': timestamp(), 'summary': str(run_dir / 'summary.json'),
        })
        print('DIAGNOSTIC_COMPLETE ' + json.dumps(summary, ensure_ascii=False), flush=True)
    except BaseException as error:
        write_json(run_dir / 'status.json', {
            'state': 'failed', 'updated_at': timestamp(), 'error_type': type(error).__name__,
            'error': str(error), 'traceback': traceback.format_exc(),
        })
        raise


if __name__ == '__main__':
    main()
