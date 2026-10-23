"""Optional AdaFE V1 diagnostics; legacy runs do not instantiate this reporter."""

import csv
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import torch


MATCHED_FIELDS = (
    'random_seed', 'data', 'root_path', 'data_path', 'features', 'target',
    'freq', 'seq_len', 'label_len', 'pred_len', 'cycle', 'use_revin',
    'gtr_cycle_len', 'gtr_gate_init', 'enc_in', 'dec_in', 'c_out',
    'd_model', 'n_heads', 'e_layers', 'd_layers', 'd_ff', 'factor', 'patch_len',
    'dropout', 'head_dropout', 'embed', 'activation', 'output_attention',
    'num_workers', 'itr', 'train_epochs', 'batch_size', 'patience',
    'learning_rate', 'loss', 'lradj', 'pct_start', 'use_amp', 'max_grad_norm',
)


def baseline_config(args):
    config = {key: getattr(args, key, None) for key in MATCHED_FIELDS}
    # Disabled clipping has the same behavior and hash as the legacy schema.
    if config['max_grad_norm'] in (None, 0.0):
        config.pop('max_grad_norm')
    return config


def config_hash(config):
    encoded = json.dumps(config, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False) + '\n',
                    encoding='utf-8')


def adafe_setting_suffix(config):
    max_grad_norm = float(getattr(config, 'max_grad_norm', 0.0))
    clip_suffix = '_clip{:g}'.format(max_grad_norm) if max_grad_norm > 0 else ''
    if getattr(config, 'use_adafe', False):
        suffix = '_adafe1_{}_fd{}_pd{}_k{}_tau{:g}_g{:g}'.format(
            getattr(config, 'adafe_mode', 'hard'),
            getattr(config, 'adafe_fourier_degree', 2),
            getattr(config, 'adafe_poly_degree', 2),
            getattr(config, 'adafe_kernel', 5),
            getattr(config, 'adafe_tau', 1.0),
            getattr(config, 'adafe_gamma_init', 0.05),
        )
        reparam = int(getattr(config, 'adafe_constant_reparam', True))
        proxy = int(getattr(config, 'adafe_proxy_normalize', True))
        lr_scale = getattr(config, 'adafe_scorer_lr_scale', 1.0)
        if reparam or proxy or lr_scale != 1.0:
            suffix += '_cr{}_ps{}_slr{:g}'.format(reparam, proxy, lr_scale)
        return suffix + clip_suffix
    return ('_adafe0' if getattr(config, 'adafe_report_dir', '') else '') + clip_suffix


def unwrap(model):
    return model.module if hasattr(model, 'module') else model


def gradient_group(parameters):
    parameters = list(parameters)
    gradients = [p.grad.detach() for p in parameters if p.grad is not None]
    if not gradients:
        return {'norm': None, 'missing': len(parameters), 'finite': True}
    squares = torch.stack([g.float().square().sum() for g in gradients])
    norm = float(squares.sum().sqrt())
    return {'norm': norm, 'missing': len(parameters) - len(gradients),
            'finite': math.isfinite(norm)}


def diagnostic_snapshot(model):
    model = unwrap(model)
    adapter = getattr(model, 'adafe_adapter', None)
    named = list(model.named_parameters())
    gtr = gradient_group(p for n, p in named
                         if n.startswith('GTR.') or n in {'Q', 'gamma'})
    backbone = gradient_group(p for n, p in named
                              if n.startswith(('en_embedding.', 'ex_embedding.',
                                               'encoder.', 'head.')))
    result = {
        'gtr_grad_norm': gtr['norm'],
        'backbone_grad_norm': backbone['norm'],
        'gradients_finite': gtr['finite'] and backbone['finite'],
        'mask_active_ratio': None, 'gamma': None,
        'scorer_grad_norm': None, 'scorer_missing_grad_count': None,
        'scorer_parameter_count': None, 'conv_grad_norm': None,
        'gamma_grad_norm': None, 'score_std': None, 'proxy_scale': None,
        'proxy_saturated_fraction': None,
    }
    if adapter is not None:
        scorer_parameters = list(adapter.adafe.score.parameters())
        scorer = gradient_group(scorer_parameters)
        conv = gradient_group([adapter.adafe.weight])
        gate = gradient_group([adapter.gamma])
        ratio = adapter.adafe.last_active_ratio
        for key, attribute in (('score_std', 'last_score_std'),
                               ('proxy_scale', 'last_proxy_scale'),
                               ('proxy_saturated_fraction', 'last_proxy_saturated_fraction')):
            value = getattr(adapter.adafe, attribute, None)
            result[key] = None if value is None else float(value)
        result.update({
            'mask_active_ratio': None if ratio is None else float(ratio),
            'gamma': float(adapter.gamma.detach()),
            'scorer_grad_norm': scorer['norm'],
            'scorer_missing_grad_count': scorer['missing'],
            'scorer_parameter_count': len(scorer_parameters),
            'conv_grad_norm': conv['norm'], 'gamma_grad_norm': gate['norm'],
            'gradients_finite': (result['gradients_finite'] and scorer['finite']
                                 and conv['finite'] and gate['finite']),
        })
    return result


class AdaFEReporter:
    """Report only at requested intervals; never add terms to the loss."""

    def __init__(self, args, model, device):
        self.args = args
        self.model = model
        self.device = device
        self.directory = Path(args.adafe_report_dir)
        if getattr(args, 'is_training', 1) and (self.directory / 'metrics.json').exists():
            raise FileExistsError('Completed AdaFE report exists; use a new run directory.')
        self.directory.mkdir(parents=True, exist_ok=True)
        self.interval = args.adafe_log_interval
        self.history = []
        self.diagnostics = []
        self.last_diagnostic = {}
        self.parameter_total = sum(p.numel() for p in model.parameters())
        adapter = getattr(unwrap(model), 'adafe_adapter', None)
        self.parameter_added = (sum(p.numel() for p in adapter.parameters())
                                if adapter is not None else 0)
        if adapter is not None:
            adapter.adafe.capture_diagnostics = True
        self.version = ('A1_hard' if args.adafe_mode == 'hard' else 'A2_ste'
                        ) if getattr(args, 'use_adafe', False) else 'B0'
        self.baseline_hash = config_hash(baseline_config(args))
        self.commit = git_commit()
        write_json(self.directory / 'config.json', {
            'args': vars(args), 'baseline_config_hash': self.baseline_hash,
            'git_commit': self.commit, 'version': self.version,
            'scorer_gradient_semantics': ('absent_by_original_hard_comparison'
                                          if self.version == 'A1_hard'
                                          else 'ste_surrogate' if adapter else None),
            'threshold_scope': 'all endogenous patch tokens in the process batch',
            'validation_shuffle': False, 'validation_drop_last': False,
            'validation_aggregation': 'target_element_weighted_mean',
        })

    def start(self, setting):
        if (self.directory / 'metrics.json').exists():
            raise FileExistsError('Completed AdaFE report exists; use a new run directory.')
        self.setting = setting
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)

    def after_backward(self, epoch, step, loss):
        if step != 1 and step % self.interval != 0:
            return
        row = {'epoch': epoch, 'step': step, 'loss': float(loss.detach())}
        row.update(diagnostic_snapshot(self.model))
        if not math.isfinite(row['loss']) or not row['gradients_finite']:
            raise FloatingPointError('Non-finite AdaFE loss/gradient diagnostics.')
        self.last_diagnostic = row
        self.diagnostics.append(row)
        with (self.directory / 'gate_diagnostics.csv').open(
                'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(self.diagnostics)
        print('AdaFE diagnostics: ' + json.dumps(row, sort_keys=True))

    def record_epoch(self, epoch, seconds, train_loss, val_loss):
        self.history.append({'epoch': epoch, 'epoch_seconds': seconds,
                             'train_mse': float(train_loss),
                             'val_mse': float(val_loss)})
        write_json(self.directory / 'training_history.json', self.history)

    def save_checkpoint(self, checkpoint):
        target = self.directory / 'checkpoint.pth'
        if Path(checkpoint).resolve() != target.resolve():
            shutil.copy2(checkpoint, target)

    def finish(self, result):
        state = diagnostic_snapshot(self.model)
        diagnostic = self.last_diagnostic
        metrics = {
            'stage': 'formal', 'dataset': Path(self.args.data_path).stem,
            'seq_len': self.args.seq_len, 'pred_len': self.args.pred_len,
            'cycle_len': self.args.gtr_cycle_len, 'seed': self.args.random_seed,
            'version': self.version, 'mse': result['standardized']['mse'],
            'mae': result['standardized']['mae'],
            'best_epoch': (min(self.history, key=lambda r: r['val_mse'])['epoch']
                           if self.history else None),
            'param_total': self.parameter_total, 'param_added': self.parameter_added,
            'peak_cuda_mb': (torch.cuda.max_memory_allocated(self.device) / 1024**2
                             if self.device.type == 'cuda' else None),
            'epoch_seconds': ([row['epoch_seconds'] for row in self.history]),
            'mask_active_ratio': state['mask_active_ratio'],
            'gamma_final': state['gamma'],
            'score_std': state['score_std'],
            'proxy_scale': state['proxy_scale'],
            'proxy_saturated_fraction': state['proxy_saturated_fraction'],
            'score_diagnostic_source': 'last forward pass',
            'scorer_grad_norm': diagnostic.get('scorer_grad_norm'),
            'scorer_missing_grad_count': diagnostic.get('scorer_missing_grad_count'),
            'conv_grad_norm': diagnostic.get('conv_grad_norm'),
            'gtr_grad_norm': diagnostic.get('gtr_grad_norm'),
            'backbone_grad_norm': diagnostic.get('backbone_grad_norm'),
            'gradient_source': 'last sampled training step before best checkpoint reload',
            'baseline_config_hash': self.baseline_hash, 'git_commit': self.commit,
            'setting': getattr(self, 'setting', result['setting']),
            'metric_space': 'standardized',
        }
        write_json(self.directory / 'metrics.json', metrics)
