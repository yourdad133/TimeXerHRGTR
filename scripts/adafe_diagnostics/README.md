# AdaFE GPU 排查脚本

2026-10-12 已清理旧 reports。历史诊断、配置及指标归档在 `X:/TEST/maintenance_archives/20261012_reports_cleanup/records/`，参见 [清理记录](../../docs/reports-cleanup-20261012.md)。以下输出目录用于新运行。

本文件夹保存本次新增的排查脚本。训练输出单独保存在 `reports/adafe_diagnostics/<运行名>/`，每组都有配置、初始化哈希、窗口索引哈希、逐轮记录、状态、最佳权重和最终结果。

## 本次运行

输出目录：`reports/adafe_diagnostics/20261011_173843`。

- 第一阶段：24 组受控训练。
- 常数基补充阶段：7 组，包括 ETTh2 第三个种子。
- 打分器学习率补充阶段：8 组，包括 Electricity 的同预算基线。
- 全数据检查：ETTh1、ETTh2、ETTm1、ETTm2、Weather、Electricity、Traffic、PEMS03、PEMS04、PEMS07、PEMS08；每个数据集进行 baseline/hard/ste/stable 三次真实更新，共 44 个检查组合、132 次更新。检查损失不能当成预测准确率基准。
- 9 项 GPU 合约测试验证原始 hard/STE 前向与梯度、GTR/骨干初始化、阈值作用范围、固定门控、重参数化、学习率分组及指标聚合。

## 远程运行

必须从项目配置的 SSH 主机运行，使用 tslib。入口会核对主机、项目目录、Python 和 CUDA。

第一阶段使用一个全新的输出目录：

```bash
bash scripts/adafe_diagnostics/run_diagnostics.sh reports/adafe_diagnostics/<unique-run-name>
```

如需复核两个补充阶段，将对应计划复制到该输出目录，再运行：

```bash
cp scripts/adafe_diagnostics/constant_reparam_plan.json reports/adafe_diagnostics/<unique-run-name>/
bash scripts/adafe_diagnostics/run_constant_reparam.sh reports/adafe_diagnostics/<unique-run-name>

cp scripts/adafe_diagnostics/scorer_lr_plan.json reports/adafe_diagnostics/<unique-run-name>/
bash scripts/adafe_diagnostics/run_scorer_lr.sh reports/adafe_diagnostics/<unique-run-name>

source scripts/adafe_diagnostics/_runtime.sh
export CUDA_VISIBLE_DEVICES=1
diagnostic_check_runtime
"$python_bin" scripts/adafe_diagnostics/collect_all.py --output reports/adafe_diagnostics/<unique-run-name>
```

入口使用 GPU 0、1 两个工作进程，每个进程最多使用该 GPU 35% 的 PyTorch 分配器显存，保留服务器上已有任务。每组时间上限为 1800 秒；失败有记录且不自动重试。相同输出目录中已完成的案例会被复用，改变计划后应使用新目录。

## 变体与训练范围

| 名称 | 与原始 STE 的单项变化 |
|---|---|
| baseline | 关闭 AdaFE，原始 TimeXerGTR |
| hard | 原有不可微硬阈值，用于梯度检查 |
| ste | 原始 V1 STE，作为对照 |
| gamma_fixed | 只将 AdaFE 残差系数固定为 0.05 |
| proxy_scaled | 只将代理 sigmoid 除以 detached 打分标准差，保留硬前向掩码 |
| token_threshold | 只将阈值改为每个 token 的隐藏特征均值 |
| constant_reparam | 两个零阶常数基折叠到打分器偏置，冻结对应系数切片，保持初始函数 |
| scorer_slow | 只将整个打分器的学习率日程乘 0.1，骨干/GTR/卷积/门控仍使用原日程 |
| stable | token 阈值 + 代理尺度归一化 + 固定门控；用于检查组合措施 |

ETTh2 使用完整时间划分，种子 2026/2027；baseline、ste、constant_reparam 另有种子 2028。Weather/ETTm2 使用种子 2026，固定 4096/512/1024 个训练/验证/测试窗口。Electricity 使用种子 2026、batch=8、固定 512/256/256 个窗口，baseline/ste/proxy_scaled/scorer_slow 同预算重训。

所有训练为 96 输入、96 预测、d_model=512、Adam 最大学习率 0.001、30 轮 OneCycleLR、patience=10。每个种子内骨干/GTR 初始值、窗口索引、训练打乱种子一致。验证保持固定顺序、不丢尾批，MSE/MAE 按全部目标元素聚合。只根据验证 MSE 选择最佳模型，选定后仅评估一次测试集。

Weather/ETTm2/Electricity 为诊断试验，不是完整数据集基准；小样本、多变体、不同训练预算的结果只作描述，不宣称统计显著性。

## 文件用途

- `run_suite.py`、`common.py`、`variants.py`：独立数据检查和训练。
- `checkpoint_probe.py`：旧 Electricity 权重的残差系数、分组、代理梯度干预。原 PEMS08 权重缺失时明确记录不可用。
- `validation_stability.py`：完整验证集上的随机丢尾波动，包含真实重算与固定窗口预测模拟。
- `scorer_decomposition.py`：常数基更新放大和旧 Electricity 打分的常数/动态成分。
- `historical_snapshot.py`：冻结当前旧 Electricity 日志及新完成的 STE 权重诊断。
- `test_diagnostics.py`：GPU 合约测试。
- `collect_all.py`：核对全部阶段，生成 `collected_summary.json`、`all_runs.csv`、`training_comparison.md` 和代码哈希清单。
- 三个 `*_plan.json`：各阶段训练开始前指定的案例和预算。补充阶段是在观察前一阶段后提出，不能将整个流程称为一次性预注册实验。

原模型、数据加载器及旧训练脚本保留本次开始时的版本；实验改动仅在本文件夹中生效。
