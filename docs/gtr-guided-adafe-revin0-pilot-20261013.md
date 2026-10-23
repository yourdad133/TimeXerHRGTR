# RevIN=0下的周期指导：小型配对训练
日期：2026-10-13。

本轮在两个数据集上均关闭RevIN，只切换同一TimeXerGTRGuidedAdaFE原型的周期指导开关。PEMS04得到明显收益；PEMS07的额外收益很小，尚不能认定其效果改善。

## 实验设置
- PEMS04和PEMS07各两组，从相同权重初始化训练。
- seed2026；输入96步，预测24步；6轮；d_model512；batch16。
- 均匀选择1024个训练窗口、256个验证窗口；保留训练集拟合的StandardScaler，use_revin=0。
- Adam固定主学习率3e-4，scorer倍率0.1，梯度裁剪范数1。
- 在固定远程主机10.21.25.109的tslib环境、GPU0运行；没有评估测试集。
- 新入口为 [run_norm0_pilot.py](X:/TEST/GTR-main/scripts/gtr_guided_adafe/run_norm0_pilot.py)，复用之前的训练流程；模型、数据处理代码未改动。

## 同结构的周期指导开关结果
指标为StandardScaler空间的验证误差，取验证MSE最低轮对应的MSE和MAE。

| 数据集 | 指导关闭MSE | 指导开启MSE | MSE变化 | 指导关闭MAE | 指导开启MAE | MAE变化 |
|---|---:|---:|---:|---:|---:|---:|
| PEMS04 | 0.107470 | 0.092765 | -13.68% | 0.235752 | 0.211298 | -10.37% |
| PEMS07 | 0.091502 | 0.091165 | -0.37% | 0.211257 | 0.211487 | +0.11% |

PEMS04指导关闭的最佳轮为6，指导开启为5；PEMS07两组均为6。每组实际训练6轮。

这组结果支持在PEMS04继续验证“RevIN关闭＋周期指导”。PEMS07本轮基本持平，额外MSE收益很小且MAE略差，需要更多种子才能判断是否具有稳定价值。

## 与上一轮原修复版的比较
上一轮原AdaFE、RevIN=0的验证MSE/MAE分别为：
- PEMS04：0.106681 / 0.234415。
- PEMS07：0.093539 / 0.214907。

本轮指导开启版本相对它们：
- PEMS04：MSE降低13.04%，MAE降低9.86%。
- PEMS07：MSE降低2.54%，MAE降低1.59%。

这不是只改变周期指导的比较：原修复版与新原型还存在scorer输入归一化、逐token阈值和STE尺度的差别。周期指导的独立贡献应以同结构开关表为准。PEMS07的新无指导版本已经达到0.091502，额外指导只进一步降到0.091165。

## 核验与运行状态
四组全部completed。每个数据集两组的骨干/GTR、共有AdaFE、周期条件参数初始哈希一致；训练/验证索引、shuffle及forward RNG一致。除model_id标签和guided_adafe_use_prior外，参数一致，两组use_revin均为0。

源码哈希在整个训练期间保持一致，也核对了与上一轮原修复版的初始化、数据索引和随机种子相同。启用指导时，最后一次记录的指导梯度范数：PEMS04为0.020524，PEMS07为0.081328；四个指导参数张量都有梯度。此次PEMS07的小收益不能归结为指导分支未连接。

核验与指标：
- [汇总与对照核验](X:/TEST/GTR-main/reports/gtr_guided_adafe_20261013/norm0_paired_summary.json)
- [PEMS04完整结果](X:/TEST/GTR-main/reports/gtr_guided_adafe_20261013/norm0_paired_pems04/summary.json)
- [PEMS07完整结果](X:/TEST/GTR-main/reports/gtr_guided_adafe_20261013/norm0_paired_pems07/summary.json)

各目录下runs/保存configuration.json、history.json、best.pth、result.json及runtime.json，原实验目录保留。

## 下一步的判断
本轮完成了所要求的小型训练。PEMS04值得进一步做更多种子及完整数据验证；PEMS07应保留关闭指导的同结构版本作为对照，不能据0.37%的单次MSE变化选择最终结构。当前结果仅对应预测24步、单种子和训练/验证子集，尚不是正式测试成绩。



## 完整训练稳定性补充（2026-10-13）

随后确认无周期指导、RevIN=0的PEMS07正式训练在OneCycle上升段仍出现损失突变。6轮pilot采用固定LR和梯度裁剪，不能替代正式训练稳定性验证。日志核查、独立学习率对照及代码修复见[训练失稳诊断](X:/TEST/GTR-main/docs/pems07-training-instability-20261013.md)。

