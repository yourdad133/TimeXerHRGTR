"""Continue only PEMS07/12-step training in a new directory on remote CUDA.

A source checkpoint containing only model weights cannot restore Adam or
its original RNG. The initial continuation reopens epoch 7 from the epoch-6
best weights with fresh Adam and the original OneCycle position. Subsequent
last_state.pt bundles restore complete state at completed epoch boundaries.
"""

import argparse
import ast
import contextlib
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from data_provider.data_factory import data_provider
from exp.exp_main import Exp_Main
from scripts.adafe_diagnostics.common import forward_batch, seed_all
from utils.tools import EarlyStopping, adjust_learning_rate

SCHEMA_VERSION = 1
EXPECTED_SOURCE_EPOCH = 6
EXPECTED_SOURCE_VAL = 0.055453274338860804
EXCLUDED_CONFIG_FIELDS = {'model_id', 'checkpoints', 'adafe_report_dir', 'gpu', 'devices'}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_state_sha256(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def cpu_model_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    try:
        with temporary.open('w', encoding='utf-8') as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    try:
        with temporary.open('wb') as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_source_log(path):
    text = Path(path).read_text(encoding='utf-8')
    line = next((line for line in text.splitlines() if line.startswith('Namespace(')), None)
    if line is None:
        raise ValueError('Source log has no Namespace configuration')
    call = ast.parse(line, mode='eval').body
    if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name)
            or call.func.id != 'Namespace' or call.args
            or any(keyword.arg is None for keyword in call.keywords)):
        raise ValueError('Expected a literal Namespace configuration')
    config = {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords}
    steps = re.findall(r'Epoch:\s*\d+,\s*Steps:\s*(\d+)\s*\|', text)
    if not steps or len(set(steps)) != 1:
        raise ValueError('Source log must record one consistent train_steps value')
    return config, int(steps[0])


def validate_task(config):
    expected = dict(model='TimeXerGTR', data='PEMS', features='M', seq_len=96,
                    pred_len=12, use_revin=0, use_adafe=1, adafe_mode='ste',
                    random_seed=2026, batch_size=16, learning_rate=3e-4,
                    lradj='TST', train_epochs=30, pct_start=.3, patience=5)
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f'Source task mismatch: {name}={config.get(name)!r}, expected {value!r}')
    if Path(config.get('data_path', '')).name != 'PEMS07.npz':
        raise ValueError('Only PEMS07.npz can be continued by this entry point')
    if (config.get('guided_adafe_use_prior', 0) or config.get('max_grad_norm', 0.) != 0.
            or config.get('use_multi_gpu', False) or config.get('use_amp', False)):
        raise ValueError('Preserve the unguided, unclipped, non-AMP single-GPU task')


def training_contract(args):
    return {name: value for name, value in vars(args).items()
            if name not in EXCLUDED_CONFIG_FIELDS}


def current_source_hashes():
    files = [Path(__file__), ROOT / 'scripts/adafe_diagnostics/common.py']
    for directory in ('models', 'layers', 'exp', 'utils', 'data_provider'):
        files.extend((ROOT / directory).rglob('*.py'))
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in sorted(set(files))}


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    if len(state['torch_cuda']) != torch.cuda.device_count():
        raise ValueError('Visible CUDA device count differs from the saved RNG state')
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state['torch_cuda']])


def advance_fresh_scheduler(optimizer, scheduler, completed_steps):
    if optimizer.state or any(parameter.grad is not None
                              for group in optimizer.param_groups for parameter in group['params']):
        raise ValueError('Weights continuation requires empty Adam state and no gradients')
    # Public APIs restore LR/beta1 only; Adam sees no gradients and creates no state.
    for unused in range(completed_steps):
        optimizer.step()
        scheduler.step()
    if optimizer.state:
        raise AssertionError('Scheduler positioning unexpectedly created Adam state')
    assert_scheduler_position(optimizer, scheduler, completed_steps)


def assert_scheduler_position(optimizer, scheduler, completed_steps):
    if scheduler.last_epoch != completed_steps:
        raise ValueError(f'OneCycle position {scheduler.last_epoch} != {completed_steps}')
    for group, expected_lr in zip(optimizer.param_groups, scheduler.get_last_lr()):
        if not math.isclose(group['lr'], expected_lr, rel_tol=1e-12, abs_tol=1e-16):
            raise ValueError('Adam LR differs from the restored OneCycle position')


class ContinuationExperiment(Exp_Main):
    def _acquire_device(self):
        # Preserve the GPU selected by CUDA_VISIBLE_DEVICES in the launching shell.
        return torch.device('cuda', self.args.gpu)


class AtomicEarlyStopping(EarlyStopping):
    def __init__(self, patience, model, baseline_val, baseline_epoch):
        super().__init__(patience=patience, verbose=True)
        self.best_score = -float(baseline_val)
        self.val_loss_min = float(baseline_val)
        self.best_epoch = int(baseline_epoch)
        self.current_epoch = int(baseline_epoch)
        self.best_model = cpu_model_state(model)

    def save_checkpoint(self, val_loss, model, path):
        self.best_model = cpu_model_state(model)
        # Publish best.pth only after the complete epoch bundle is committed.
        self.val_loss_min = float(val_loss)
        self.best_epoch = self.current_epoch
        if self.verbose:
            print(f'Validation best updated: epoch={self.best_epoch} MSE={val_loss:.9f}', flush=True)

    def saved_state(self):
        return dict(patience=self.patience, delta=self.delta, counter=self.counter,
                    best_score=self.best_score, val_loss_min=self.val_loss_min,
                    early_stop=self.early_stop, best_epoch=self.best_epoch)

    def restore_state(self, state, best_model):
        for name, value in state.items():
            setattr(self, name, value)
        self.best_model = best_model


def make_bundle(exp, optimizer, scheduler, epoch, stopper, history, provenance,
                source_args, data_contract, code_hashes):
    return dict(schema_version=SCHEMA_VERSION, completed_epoch=int(epoch),
                checkpoint_boundary='completed_epoch', continuation_kind='weights_continuation',
                model=cpu_model_state(exp.model), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), early_stopping=stopper.saved_state(),
                best_model=stopper.best_model, history=copy.deepcopy(history), rng=capture_rng(),
                source_args=source_args, effective_args=vars(exp.args).copy(),
                training_contract=training_contract(exp.args), data_contract=data_contract,
                code_hashes=code_hashes, provenance=provenance)


def restore_bundle(bundle, exp, optimizer, scheduler, stopper, expected_data, expected_code):
    if bundle.get('schema_version') != SCHEMA_VERSION or bundle.get('checkpoint_boundary') != 'completed_epoch':
        raise ValueError('Unsupported or incomplete continuation state')
    if bundle['training_contract'] != training_contract(exp.args):
        raise ValueError('Training configuration differs from the saved full-state bundle')
    if bundle['data_contract'] != expected_data or bundle['code_hashes'] != expected_code:
        raise ValueError('Data or source code differs from the saved full-state bundle')
    exp.model.load_state_dict(bundle['model'], strict=True)
    optimizer.load_state_dict(bundle['optimizer'])
    scheduler.load_state_dict(bundle['scheduler'])
    stopper.restore_state(bundle['early_stopping'], bundle['best_model'])
    assert_scheduler_position(optimizer, scheduler,
                              int(bundle['completed_epoch']) * expected_data['train_steps'])
    # The caller restores RNG last, after initialization and baseline validation.


def commit_boundary(output, bundle, status=None):
    """The complete bundle is authoritative; delay signals during its commit."""
    output = Path(output)
    if status is not None:
        status.defer_signals += 1
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    try:
        atomic_json(output / 'training_history.json', bundle['history'])
        atomic_torch_save(output / 'last_state.pt', bundle)
        if status is not None:
            stopping = bundle['early_stopping']
            status.write(stage='publishing_best', last_completed_epoch=bundle['completed_epoch'],
                         pending_epoch=None, pending_batch=None, best_epoch=stopping['best_epoch'],
                         best_validation_mse=stopping['val_loss_min'],
                         early_stopping_counter=stopping['counter'])
        atomic_torch_save(output / 'best.pth', bundle['best_model'])
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        if status is not None:
            status.defer_signals -= 1
            status.flush_deferred_signal()


class TrainingInterrupted(BaseException):
    def __init__(self, signum):
        self.signum = int(signum)
        self.exit_code = 128 + self.signum
        super().__init__(f'Received {signal.Signals(signum).name}')


class RunStatus:
    def __init__(self, output):
        self.output = Path(output)
        self.defer_signals = 0
        self.pending_signal = None
        self.value = dict(state='starting', pid=os.getpid(), last_completed_epoch=None,
                          pending_epoch=None, pending_batch=None, stage='initialization')
        self.write()

    def write(self, **updates):
        self.value.update(updates)
        self.value['updated_unix_seconds'] = time.time()
        atomic_json(self.output / 'status.json', self.value)

    def interrupted(self, signum):
        self.write(state='interrupted', termination_reason=signal.Signals(signum).name,
                   exit_code=128 + int(signum),
                   resumable_bundle=str(self.output / 'last_state.pt')
                   if (self.output / 'last_state.pt').exists() else None)

    def handle_signal(self, signum, unused_frame):
        # CUDA/native threads can receive process signals despite a Python-thread mask.
        if self.defer_signals:
            self.pending_signal = int(signum)
            return
        self.interrupted(signum)
        raise TrainingInterrupted(signum)

    def flush_deferred_signal(self):
        if not self.defer_signals and self.pending_signal is not None:
            signum, self.pending_signal = self.pending_signal, None
            self.handle_signal(signum, None)


class Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log

    def write(self, text):
        self.log.write(text)
        self.log.flush()
        return self.original.write(text)

    def flush(self):
        self.log.flush()
        self.original.flush()


def runtime_preflight(gpu):
    expected_python = Path('/home/ouyangguanghong/miniconda3/envs/tslib/bin/python')
    expected_project = Path('/home/ouyangguanghong/projects/TEST/GTR-main')
    if (ROOT != expected_project or Path.cwd().resolve() != expected_project
            or Path(sys.executable).resolve() != expected_python.resolve()):
        raise RuntimeError('Run through SSH from the configured project with tslib Python')
    addresses = subprocess.check_output(['hostname', '-I'], text=True).split()
    connection = os.environ.get('SSH_CONNECTION', '').split()
    if '10.21.25.109' not in addresses or len(connection) != 4 or connection[-2:] != ['10.21.25.109', '22']:
        raise RuntimeError('The configured SSH host 10.21.25.109 is required')
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total',
                                         '--format=csv,noheader'], text=True)
    if not torch.cuda.is_available() or not 0 <= gpu < torch.cuda.device_count():
        raise RuntimeError('The selected remote CUDA GPU is unavailable')
    if torch.cuda.device_count() != 1 or gpu != 0:
        raise RuntimeError('Select one physical GPU with CUDA_VISIBLE_DEVICES, using local --gpu 0')
    info = dict(host=subprocess.check_output(['hostname'], text=True).strip(), addresses=addresses,
                project=str(ROOT), python=sys.executable, torch=torch.__version__, cuda=True,
                gpu_local_index=gpu, visible_gpus=os.environ.get('CUDA_VISIBLE_DEVICES'),
                gpu_inventory=inventory)
    print('RUNTIME ' + json.dumps(info), flush=True)
    return info


def parse_args(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-log', type=Path)
    parser.add_argument('--source-checkpoint', type=Path)
    parser.add_argument('--source-history', type=Path)
    parser.add_argument('--resume-bundle', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    options = parser.parse_args(arguments)
    first = [options.source_log, options.source_checkpoint, options.source_history]
    if (options.resume_bundle and any(first)) or (not options.resume_bundle and not all(first)):
        parser.error('Use all three --source-* paths, or only --resume-bundle')
    if not options.output.is_absolute():
        parser.error('--output must be an absolute remote project path')
    return options


def prepare_output(output):
    output = output.resolve()
    if ROOT not in output.parents:
        raise ValueError('The new output directory must be inside the remote project')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new, empty continuation output directory')
    output.mkdir(parents=True, exist_ok=True)
    return output


def source_baseline(options):
    config, source_steps = parse_source_log(options.source_log)
    validate_task(config)
    history = json.loads(options.source_history.read_text())
    if not history:
        raise ValueError('The source training history is empty')
    best = min(history, key=lambda row: row['validation_mse_standardized'])
    if best['epoch'] != EXPECTED_SOURCE_EPOCH or not math.isclose(
            best['validation_mse_standardized'], EXPECTED_SOURCE_VAL, rel_tol=0., abs_tol=1e-10):
        raise ValueError('The source history does not identify the expected epoch-6 best weights')
    history = [row for row in history if int(row['epoch']) <= EXPECTED_SOURCE_EPOCH]
    if [int(row['epoch']) for row in history] != list(range(1, EXPECTED_SOURCE_EPOCH + 1)):
        raise ValueError('The source history must contain completed epochs 1 through 6')
    provenance = dict(source_log=str(options.source_log.resolve()),
                      source_checkpoint=str(options.source_checkpoint.resolve()),
                      source_history=str(options.source_history.resolve()),
                      source_sha256={name: file_sha256(path) for name, path in
                                     [('log', options.source_log), ('checkpoint', options.source_checkpoint),
                                      ('history', options.source_history)]},
                      source_best_epoch=EXPECTED_SOURCE_EPOCH, source_best_val=EXPECTED_SOURCE_VAL,
                      original_optimizer_available=False, original_rng_available=False,
                      partial_epoch_7_step_600_available=False,
                      initial_recovery='epoch-6 weights, fresh Adam, reconstructed OneCycle position')
    state = torch.load(options.source_checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not state or not all(torch.is_tensor(value) for value in state.values()):
        raise ValueError('The initial source checkpoint must contain only a model state_dict')
    return config, source_steps, history, provenance, state


def effective_args(source_args, output, gpu):
    values = source_args.copy()
    values['root_path'] = str((ROOT / values['root_path']).resolve())
    values.update(model_id='PEMS07_resume12_' + re.sub(r'[^A-Za-z0-9_-]', '_', output.name)[:48],
                  checkpoints=str(output / 'checkpoints'), adafe_report_dir=str(output / 'adafe_diagnostics'),
                  adafe_log_interval=100, gpu=gpu, use_gpu=True, use_multi_gpu=False,
                  max_grad_norm=0.)
    return argparse.Namespace(**values)


def data_fingerprint(args, datasets, loaders):
    data_file = (Path(args.root_path) / args.data_path).resolve()
    scaler = datasets['train'].scaler
    scaler_digest = hashlib.sha256(scaler.mean_.tobytes() + scaler.scale_.tobytes()).hexdigest()
    return dict(data_file=str(data_file), data_sha256=file_sha256(data_file),
                scaler_sha256=scaler_digest, batch_size=args.batch_size,
                sample_counts={name: len(dataset) for name, dataset in datasets.items()},
                train_steps=len(loaders['train']))


def reporter_history(history):
    return [dict(epoch=row['epoch'], epoch_seconds=row.get('epoch_seconds'),
                 train_mse=row['train_mse_standardized'], val_mse=row['validation_mse_standardized'])
            for row in history]


def execute(options, output, status):
    runtime = runtime_preflight(options.gpu)
    atomic_json(output / 'runtime.json', runtime)
    resumed = options.resume_bundle is not None
    if resumed:
        bundle = torch.load(options.resume_bundle, map_location='cpu', weights_only=False)
        source_args, provenance = bundle['source_args'], bundle['provenance']
        validate_task(source_args)
        source_steps = int(bundle['data_contract']['train_steps'])
        history = copy.deepcopy(bundle['history'])
        completed_epoch = int(bundle['completed_epoch'])
        state = bundle['model']
        baseline_expected = float(history[-1]['validation_mse_standardized'])
        provenance = dict(provenance, resumed_bundle=str(options.resume_bundle.resolve()),
                          resumed_bundle_sha256=file_sha256(options.resume_bundle))
    else:
        source_args, source_steps, history, provenance, state = source_baseline(options)
        completed_epoch = EXPECTED_SOURCE_EPOCH
        baseline_expected = EXPECTED_SOURCE_VAL
    args = effective_args(source_args, output, options.gpu)
    print('Args in experiment:')
    print(args, flush=True)
    print('Continuation mode: ' + ('full_state_resume' if resumed else 'weights_continuation'), flush=True)
    if not resumed:
        print('Original Adam/RNG and epoch-7 batch-600 state are absent; reopen epoch 7 with fresh Adam.', flush=True)
    seed_all(args.random_seed)
    exp = ContinuationExperiment(args)
    initial_hash = tensor_state_sha256(exp.model.state_dict())
    exp.model.load_state_dict(state, strict=True)
    loaded_hash = tensor_state_sha256(exp.model.state_dict())
    if loaded_hash != tensor_state_sha256(state):
        raise ValueError('Strict checkpoint loading changed the source model tensor state')
    datasets, loaders = {}, {}
    for flag in ('train', 'val', 'test'):
        datasets[flag], loaders[flag] = data_provider(args, flag)
    data_contract = data_fingerprint(args, datasets, loaders)
    train_steps = data_contract['train_steps']
    if train_steps != source_steps:
        raise ValueError(f'Current train_steps {train_steps} differs from source {source_steps}')
    if train_steps < 1:
        raise ValueError('The training loader is empty')
    code_hashes = current_source_hashes()
    optimizer = exp._select_optimizer()
    scheduler = exp._select_scheduler(optimizer, train_steps)
    stopper = AtomicEarlyStopping(args.patience, exp.model, baseline_expected, completed_epoch)
    if resumed:
        restore_bundle(bundle, exp, optimizer, scheduler, stopper, data_contract, code_hashes)
    else:
        advance_fresh_scheduler(optimizer, scheduler, completed_epoch * train_steps)
        recorded_groups = history[-1].get('optimizer_groups', [])
        for group, recorded in zip(optimizer.param_groups, recorded_groups):
            if not math.isclose(group['lr'], recorded['learning_rate_after_training'], rel_tol=1e-12):
                raise ValueError('Reconstructed LR does not match the source completed epoch')
            if tuple(group['betas']) != tuple(recorded['adam_betas_after_training']):
                raise ValueError('Reconstructed Adam betas do not match the source completed epoch')
    criterion = exp._select_criterion()
    status.write(stage='baseline_validation', pending_epoch=None, pending_batch=None)
    baseline_val = exp.vali(datasets['val'], loaders['val'], criterion)
    if not math.isfinite(baseline_val) or not math.isclose(baseline_val, baseline_expected, rel_tol=0., abs_tol=1e-6):
        raise ValueError(f'Baseline validation mismatch: actual {baseline_val:.12f}, expected {baseline_expected:.12f}')
    atomic_json(output / 'baseline_validation.json', dict(validation_mse=baseline_val,
                expected_validation_mse=baseline_expected, absolute_tolerance=1e-6,
                loaded_model_sha256=loaded_hash, initialized_model_sha256=initial_hash,
                source_model_sha256=tensor_state_sha256(state), completed_epoch=completed_epoch,
                adam_state_entries=len(optimizer.state),
                optimizer_groups=[dict(lr=group['lr'], betas=list(group['betas']),
                                       lr_scale=group.get('lr_scale', 1.)) for group in optimizer.param_groups]))
    setting = args.model_id + '_TimeXerGTR_PEMS_ftM_sl96_pl12_cycle288_seed2026_adafe_ste_revin0'
    exp.adafe_reporter.start(setting)
    exp.adafe_reporter.history = reporter_history(history)
    atomic_json(output / 'config.json', dict(args=vars(args), source_args=source_args,
                provenance=provenance, continuation_mode='full_state_resume' if resumed else 'weights_continuation',
                code_hashes=code_hashes, data_contract=data_contract, output=str(output), setting=setting))
    if resumed:
        restore_rng(bundle['rng'])
    # Commit an initial completed boundary so an interrupted epoch 7 is resumable.
    commit_boundary(output, make_bundle(exp, optimizer, scheduler, completed_epoch,
                    stopper, history, provenance, source_args, data_contract, code_hashes), status)
    status.write(state='running', stage='epoch_boundary', last_completed_epoch=completed_epoch,
                 best_epoch=stopper.best_epoch, best_validation_mse=stopper.val_loss_min,
                 continuation_mode='full_state_resume' if resumed else 'weights_continuation')
    for epoch in range(completed_epoch + 1, args.train_epochs + 1):
        if stopper.early_stop:
            break
        status.write(stage='training', pending_epoch=epoch, pending_batch=0)
        exp.model.train()
        losses = []
        epoch_seconds = time.time()
        optimizer_rows = [dict(group_index=index, lr_scale=group.get('lr_scale', 1.),
                          learning_rate_first_step=group['lr'], learning_rate_min=group['lr'],
                          learning_rate_max=group['lr'], adam_beta1_first_step=group['betas'][0],
                          adam_beta1_min=group['betas'][0], adam_beta1_max=group['betas'][0])
                          for index, group in enumerate(optimizer.param_groups)]
        for step, batch in enumerate(loaders['train'], 1):
            status.value['pending_batch'] = step
            for group, row in zip(optimizer.param_groups, optimizer_rows):
                lr, beta1 = float(group['lr']), float(group['betas'][0])
                row.update(learning_rate_last_step=lr, adam_beta1_last_step=beta1,
                           learning_rate_min=min(row['learning_rate_min'], lr),
                           learning_rate_max=max(row['learning_rate_max'], lr),
                           adam_beta1_min=min(row['adam_beta1_min'], beta1),
                           adam_beta1_max=max(row['adam_beta1_max'], beta1))
            optimizer.zero_grad()
            prediction, truth = forward_batch(exp.model, batch, exp.device)
            loss = criterion(prediction, truth)
            loss_value = float(loss.detach())
            if not math.isfinite(loss_value):
                raise FloatingPointError('Non-finite continuation training loss')
            losses.append(loss_value)
            exp._backward_step(loss, optimizer, epoch, step)
            adjust_learning_rate(optimizer, scheduler, epoch, args, printout=False)
            scheduler.step()
            if step % 100 == 0:
                print(f'\titers: {step}, epoch: {epoch} | loss: {loss_value:.7f}', flush=True)
                status.write()
        epoch_seconds = time.time() - epoch_seconds
        status.write(stage='validation', pending_batch=None)
        validation = exp.vali(datasets['val'], loaders['val'], criterion)
        if not math.isfinite(validation):
            raise FloatingPointError('Non-finite continuation validation loss')
        for group, row in zip(optimizer.param_groups, optimizer_rows):
            row.update(learning_rate_after_training=float(group['lr']),
                       adam_betas_after_training=[float(beta) for beta in group['betas']])
        train_loss = float(np.mean(losses))
        print(f'Epoch: {epoch}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {validation:.7f}', flush=True)
        stopper.current_epoch = epoch
        stopper(validation, exp.model, str(output))
        history.append(dict(epoch=epoch, train_mse_standardized=train_loss,
                       validation_mse_standardized=validation, epoch_seconds=epoch_seconds,
                       optimizer_groups=optimizer_rows, learning_rate_schedule=args.lradj,
                       max_grad_norm=args.max_grad_norm, best_epoch=stopper.best_epoch,
                       early_stopping_counter=stopper.counter, early_stopped=stopper.early_stop))
        exp.adafe_reporter.record_epoch(epoch, epoch_seconds, train_loss, validation)
        status.write(stage='saving_completed_epoch')
        commit_boundary(output, make_bundle(exp, optimizer, scheduler, epoch, stopper,
                        history, provenance, source_args, data_contract, code_hashes), status)
        completed_epoch = epoch
        status.write(stage='epoch_boundary', last_completed_epoch=completed_epoch,
                     pending_epoch=None, pending_batch=None, best_epoch=stopper.best_epoch,
                     best_validation_mse=stopper.val_loss_min, early_stopping_counter=stopper.counter)
        print(f'Updating learning rate to {scheduler.get_last_lr()[0]}', flush=True)
        if stopper.early_stop:
            print('Early stopping', flush=True)
            break
    status.write(stage='final_test', pending_epoch=None, pending_batch=None)
    exp.model.load_state_dict(stopper.best_model, strict=True)
    exp.adafe_reporter.save_checkpoint(str(output / 'best.pth'))
    # The new directory is inside the project; absolute data paths let Exp_Main.test
    # keep all result.txt/results/test_results artifacts inside this isolated output.
    original_directory = Path.cwd()
    try:
        os.chdir(output)
        exp.test(setting)
    finally:
        os.chdir(original_directory)
    atomic_json(output / 'completion.json', dict(completed_epoch=completed_epoch,
                best_epoch=stopper.best_epoch, best_validation_mse=stopper.val_loss_min,
                stop_reason='early_stopping' if stopper.early_stop else 'global_epoch_budget',
                best_checkpoint_sha256=file_sha256(output / 'best.pth'),
                final_metrics=str(output / 'adafe_diagnostics/metrics.json'),
                source_optimizer_was_missing=True, source_partial_epoch_7_was_not_restored=True))
    status.write(state='completed', stage='complete', exit_code=0,
                 stop_reason='early_stopping' if stopper.early_stop else 'global_epoch_budget')
    return 0


def main(arguments=None):
    options = parse_args(arguments)
    output = prepare_output(options.output)
    with (output / 'train.log').open('w', encoding='utf-8') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            status = RunStatus(output)
            previous_handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
            for signum in previous_handlers:
                signal.signal(signum, status.handle_signal)
            try:
                return execute(options, output, status)
            except TrainingInterrupted as interrupted:
                print(str(interrupted), flush=True)
                return interrupted.exit_code
            except Exception as error:
                status.write(state='failed', exit_code=1, error_type=type(error).__name__,
                             error=str(error), traceback=traceback.format_exc())
                traceback.print_exc()
                return 1
            finally:
                for signum, handler in previous_handlers.items():
                    signal.signal(signum, handler)


if __name__ == '__main__':
    raise SystemExit(main())
