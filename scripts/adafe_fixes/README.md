# AdaFE 三项修复与复核

2026-10-12 已清理旧 reports。历史配置、结果和结论归档在 `X:/TEST/maintenance_archives/20261012_reports_cleanup/records/`，参见 [清理记录](../../docs/reports-cleanup-20261012.md)。下面的命令会生成新的实验输出。

修改入口为 X 盘映射的服务器项目，所有运行限定在 10.21.25.109 的 tslib 环境。

## 生产实现

- AdaFE 默认启用 `adafe_constant_reparam=1`：cos(0x) 与 x^0 的贡献折叠到显式偏置。保留旧张量形状和初始化随机数，前向跳过两个零阶切片；真实梯度为零，Adam/AdamW 不会重复更新。`load_state_dict` 自动折叠旧权重，并可重复加载新权重。
- STE 默认启用 `adafe_proxy_normalize=1`：`sigmoid((score-mean(score))/(tau*max(detached_std(score),1e-4)))`。归一化只作用于代理梯度，保留 batch 全局硬阈值。代理采用 FP32，硬前向使用 `hard + (soft-soft.detach())`，并保留输入 dtype。
- 验证/测试 DataLoader 固定顺序、保留尾批；生产 `vali` 按预测目标元素数加权。
- `--adafe_scorer_lr_scale 0.1` 可只减慢打分器。默认仍为 1.0，因原诊断没有跨数据集统一收益；OneCycleLR 和其他日程均保留组间倍率，骨干、GTR、卷积和门控维持原倍率。
- 修改后的 AdaFE checkpoint setting 加入 `_cr*_ps*_slr*`，与旧设置分开；报告新增 score_std、代理尺度和饱和比例。

## 运行与配置

通过已配置 SSH 登录服务器后：

```bash
cd /home/ouyangguanghong/projects/TEST/GTR-main
bash scripts/adafe_fixes/run_verify.sh reports/adafe_fixes/<new-run>
```

入口先核对主机、项目、解释器、nvidia-smi 和 CUDA，再执行 30 个专属检查、11 个数据集的 33 个组合/99 次实际更新，以及计划中的 20 组匹配短训；使用 GPU 0、1，限制单进程 PyTorch 分配器显存并降低 CPU 优先级。使用新的输出目录。

在原有 `run.py` 完整训练参数中使用：

```text
--use_adafe 1 --adafe_mode ste
--adafe_constant_reparam 1 --adafe_proxy_normalize 1
--adafe_scorer_lr_scale 0.1
```

此学习率配置作为 Electricity 的后续复核候选。若只评估新默认实现，将倍率设为 1.0。

复现旧打分器和代理：

```text
--adafe_constant_reparam 0 --adafe_proxy_normalize 0 --adafe_scorer_lr_scale 1
```

验证协议仍使用修复后的完整窗口口径；要连同旧随机验证协议一起恢复，使用 Git 快照。旧诊断脚本的配置已明确设置为旧打分器/代理，原诊断报告和历史权重保留。

## 本轮证据

本轮结果位于 `reports/adafe_fixes/20261011/`，主报告 `fix_report.md`，全部结果 `summary.json`。短训统一 seed 2026、6 轮、512/257/257 个窗口，训练窗口在各原时间划分内部取样、标准化仅使用训练划分；只用验证选模，选择后测试一次。20 组均在训练前写入 plan.json，固定初始化和窗口索引。

当前回归覆盖普通 GTR 和 AdaFE。早期停止支持的测试设计及其原始日志已移入仓库外的历史存档，位于 `X:/TEST/maintenance_archives/20261012_project_cleanup/originals/`；代码原件可从维护前 Git 标签读取。

## 回退

修改前提交：`8d1b1f8976b6fc4cec5f183fe7a8e64af867547b`；标签：`adafe-before-fixes-20261011`。

在没有新的代码改动需要保留时，可从该标签另建回退分支：

```bash
git switch -c codex/adafe-rollback-20261011 adafe-before-fixes-20261011
```

或只恢复本轮修改的已有文件：

```bash
git restore --source=adafe-before-fixes-20261011 -- \
  layers/adafe_v1.py models/TimeXerGTR.py data_provider/data_factory.py \
  exp/exp_main.py run.py utils/tools.py utils/adafe_reporting.py \
  scripts/adafe_diagnostics/common.py \
  scripts/adafe_diagnostics/test_diagnostics.py tests/test_adafe_v1.py
```

Git 快照包含当时已有代码、非忽略报告及日志和原有删除状态。dataset、checkpoints、results 等既有忽略产物保留在服务器，未纳入 Git，回退代码不会删除这些产物。
