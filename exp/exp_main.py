from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, SegRNN, CycleNet, \
    iTransformer, TimeXer, TimeXerGTR, GTR, GTRDLinear, GTRPatchTST, GTRiTransformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
from utils.adafe_reporting import AdaFEReporter

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import json
import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.adafe_reporter = (
            AdaFEReporter(args, self.model, self.device)
            if getattr(args, 'adafe_report_dir', '') else None
        )

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'SegRNN': SegRNN,
            'CycleNet': CycleNet,
            'iTransformer': iTransformer,
            'TimeXer': TimeXer,
            'TimeXerGTR': TimeXerGTR,
            'GTR': GTR,
            'GTRDLinear': GTRDLinear,
            'GTRPatchTST': GTRPatchTST,
            'GTRiTransformer': GTRiTransformer
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model = self.model.module if hasattr(self.model, "module") else self.model
        scorer_scale = (float(getattr(self.args, "adafe_scorer_lr_scale", 1.0))
                        if getattr(model, "use_adafe", False) else 1.0)
        if not np.isfinite(scorer_scale) or scorer_scale <= 0:
            raise ValueError("AdaFE scorer LR scale must be finite and positive.")
        scorer_parameters = (list(model.adafe_adapter.adafe.score.parameters())
                             if scorer_scale != 1.0 else [])
        scorer_ids = {id(parameter) for parameter in scorer_parameters}
        other_parameters = [parameter for parameter in self.model.parameters()
                            if parameter.requires_grad and id(parameter) not in scorer_ids]
        groups = [{"params": other_parameters, "lr": self.args.learning_rate,
                   "lr_scale": 1.0}]
        if scorer_parameters:
            groups.append({"params": scorer_parameters,
                           "lr": self.args.learning_rate * scorer_scale,
                           "lr_scale": scorer_scale})
        return optim.Adam(groups)

    def _select_scheduler(self, model_optim, train_steps):
        # Constructing OneCycleLR changes both LR and Adam beta1 immediately.
        if self.args.lradj != 'TST':
            return None
        return lr_scheduler.OneCycleLR(
            optimizer=model_optim, steps_per_epoch=train_steps,
            pct_start=self.args.pct_start, epochs=self.args.train_epochs,
            max_lr=[self.args.learning_rate * group.get('lr_scale', 1.0)
                    for group in model_optim.param_groups],
        )

    def _backward_step(self, loss, model_optim, epoch, step, scaler=None):
        max_grad_norm = float(getattr(self.args, 'max_grad_norm', 0.0))
        if scaler is not None:
            scaler.scale(loss).backward()
            if self.adafe_reporter is not None or max_grad_norm > 0:
                scaler.unscale_(model_optim)
        else:
            loss.backward()
        # Diagnostics describe the original, unscaled gradient before clipping.
        if self.adafe_reporter is not None:
            self.adafe_reporter.after_backward(epoch, step, loss)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        if scaler is not None:
            scaler.step(model_optim)
            scaler.update()
        else:
            model_optim.step()

    def _write_training_history(self, history_path, loss_history):
        # Replace atomically so a stopped process leaves its completed epochs.
        temporary_path = history_path + '.tmp'
        with open(temporary_path, 'w', encoding='utf-8') as history_file:
            json.dump(loss_history, history_file, indent=2)
        os.replace(temporary_path, history_path)

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss, total_values = 0.0, 0
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle = batch[:5]
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model == 'TimeXerGTR':
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)
                        elif any(substr in self.args.model for substr in {'CycleNet', 'GTR'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'SegRNN', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.model == 'TimeXerGTR':
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)
                    elif any(substr in self.args.model for substr in {'CycleNet', 'GTR'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                # Weight the retained tail batch by its actual target elements.
                total_loss += float(loss) * true.numel()
                total_values += true.numel()
        self.model.train(was_training)
        if not total_values:
            raise ValueError("Validation loader contains no target values.")
        return total_loss / total_values

    def train(self, setting):
        if self.adafe_reporter is not None:
            self.adafe_reporter.start(setting)
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')


        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        loss_history = []
        history_dir = os.path.join('./results', setting)
        os.makedirs(history_dir, exist_ok=True)
        history_path = os.path.join(history_dir, 'training_history.json')

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = self._select_scheduler(model_optim, train_steps)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            epoch_optimizer_groups = [
                {
                    'group_index': group_index,
                    'lr_scale': float(group.get('lr_scale', 1.0)),
                    'learning_rate_first_step': float(group['lr']),
                    'learning_rate_min': float(group['lr']),
                    'learning_rate_max': float(group['lr']),
                    'adam_beta1_first_step': float(group['betas'][0]),
                    'adam_beta1_min': float(group['betas'][0]),
                    'adam_beta1_max': float(group['betas'][0]),
                }
                for group_index, group in enumerate(model_optim.param_groups)
            ]
            # max_memory = 0
            for i, batch in enumerate(train_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle = batch[:5]
                iter_count += 1
                for group, state in zip(model_optim.param_groups, epoch_optimizer_groups):
                    lr, beta1 = float(group['lr']), float(group['betas'][0])
                    state['learning_rate_min'] = min(state['learning_rate_min'], lr)
                    state['learning_rate_max'] = max(state['learning_rate_max'], lr)
                    state['learning_rate_last_step'] = lr
                    state['adam_beta1_min'] = min(state['adam_beta1_min'], beta1)
                    state['adam_beta1_max'] = max(state['adam_beta1_max'], beta1)
                    state['adam_beta1_last_step'] = beta1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model == 'TimeXerGTR':
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)
                        elif any(substr in self.args.model for substr in {'CycleNet', 'GTR'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'SegRNN', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.model == 'TimeXerGTR':
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)
                    elif any(substr in self.args.model for substr in {'CycleNet', 'GTR'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                self._backward_step(
                    loss, model_optim, epoch + 1, i + 1,
                    scaler=scaler if self.args.use_amp else None,
                )

                # current_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
                # max_memory = max(max_memory, current_memory)

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            epoch_training_seconds = time.time() - epoch_time
            for group, state in zip(model_optim.param_groups, epoch_optimizer_groups):
                state['learning_rate_after_training'] = float(group['lr'])
                state['adam_betas_after_training'] = [float(beta) for beta in group['betas']]
            print("Epoch: {} cost time: {}".format(epoch + 1, epoch_training_seconds))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            loss_history.append({
                "epoch": int(epoch + 1),
                "train_mse_standardized": float(train_loss),
                "validation_mse_standardized": float(vali_loss),
                "test_mse_standardized_diagnostic": float(test_loss),
                "optimizer_groups": epoch_optimizer_groups,
                "learning_rate_schedule": self.args.lradj,
                "max_grad_norm": float(getattr(self.args, 'max_grad_norm', 0.0)),
            })
            self._write_training_history(history_path, loss_history)
            if self.adafe_reporter is not None:
                self.adafe_reporter.record_epoch(
                    epoch + 1, epoch_training_seconds, train_loss, vali_loss)
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        if self.adafe_reporter is not None:
            self.adafe_reporter.save_checkpoint(best_model_path)

        self._write_training_history(history_path, loss_history)

        # print(f"Max Memory (MB): {max_memory}")

        return self.model

    def test(self, setting, test=0):

        test_data, test_loader = self._get_data(flag='test')


        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle = batch[:5]
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model == 'TimeXerGTR':
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)
                        elif any(substr in self.args.model for substr in {'CycleNet', 'GTR'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'SegRNN', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.model == 'TimeXerGTR':
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)
                    elif any(substr in self.args.model for substr in {'CycleNet', 'GTR'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                # inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()

                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)

                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                    # np.savetxt(os.path.join(folder_path, str(i) + '.txt'), pd)
                    # np.savetxt(os.path.join(folder_path, str(i) + 'true.txt'), gt)

        if self.args.test_flop:
            test_params_flop(self.model, (batch_x.shape[1], batch_x.shape[2]))
            exit()
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        # inputx = np.concatenate(inputx, axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        ### denorm ###
        # denorm_preds = np.stack([test_data.inverse_transform(pred) for pred in preds])
        # denorm_trues = np.stack([test_data.inverse_transform(true) for true in trues])

        ### denorm ###

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 不再复制（省内存，数值不变）：
        #   - metric() 只做临时数组运算，不修改入参；
        #   - StandardScaler().inverse_transform 默认 copy=True，返回新数组；
        #   - result['standardized'] 在此处已算完并存成 float，反归一化的结果会重新
        #     绑定到 preds/trues，standardized_* 之后再未被引用。
        standardized_preds = preds
        standardized_trues = trues
        norm_mae, norm_mse, norm_rmse, _, _, _, _ = metric(standardized_preds, standardized_trues)
        result = {
            "setting": setting,
            "pred_len": int(self.args.pred_len),
            "target": self.args.target,
            "test_windows": int(len(test_data)),
            "scored_target_values": int(standardized_trues.size),
            "standardized": {
                "mae": float(norm_mae),
                "mse": float(norm_mse),
                "rmse": float(norm_rmse),
            },
        }

        if getattr(self.args, 'report_denormalized_metrics', False):
            preds = test_data.inverse_transform(standardized_preds)
            trues = test_data.inverse_transform(standardized_trues)
            mae, mse, rmse, _, _, _, _ = metric(preds, trues)
            target_index = 0 if self.args.features == 'S' else len(test_data.scaler.mean_) - 1
            result['original_units'] = {
                "mae": float(mae),
                "mse": float(mse),
                "rmse": float(rmse),
            }
            result['target_scaler'] = {
                "mean": float(test_data.scaler.mean_[target_index]),
                "std": float(test_data.scaler.scale_[target_index]),
            }
            forecast_path = os.path.join(folder_path, 'test_forecasts.npz')
            np.savez_compressed(
                forecast_path,
                prediction=preds.astype(np.float32),
                truth=trues.astype(np.float32),
            )
            result['forecast_file'] = forecast_path
            metrics_path = os.path.join(folder_path, 'test_metrics.json')
            with open(metrics_path, 'w', encoding='utf-8') as metrics_file:
                json.dump(result, metrics_file, indent=2)

        if self.adafe_reporter is not None:
            self.adafe_reporter.finish(result)
        mae, mse, rmse = result.get('original_units', result['standardized']).values()
        print('mse:{}, mae:{}, rmse:{}'.format(mse, mae, rmse))
        if 'original_units' in result:
            print('standardized_mse:{}, standardized_mae:{}'.format(
                result['standardized']['mse'], result['standardized']['mae']))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse]))
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(pred_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle = batch[:5]
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(
                    batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model == 'TimeXerGTR':
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)
                        elif any(substr in self.args.model for substr in {'CycleNet', 'GTR'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'SegRNN', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.model == 'TimeXerGTR':
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)
                    elif any(substr in self.args.model for substr in {'CycleNet', 'GTR'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
