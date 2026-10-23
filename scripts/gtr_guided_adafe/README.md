# GTR-guided AdaFE：隔离原型与诊断
此目录不会改写生产 run.py 的结果表、日志或权重。模型定义位于 models/TimeXerGTRGuidedAdaFE.py；它可以直接由本目录的runner训练，无需先修改生产模型注册。

原型使用对齐的 Q 潜在周期位置条件调节 AdaFE scorer，同时归一化 scorer 输入并改用逐 token 阈值/STE尺度。Q 不被解释为真实周期分量；当前模块也不执行时间 FFT。

## 运行位置
所有命令必须在已通过SSH登录的固定主机10.21.25.109上执行，项目为 /home/ouyangguanghong/projects/TEST/GTR-main，Python为 /home/ouyangguanghong/miniconda3/envs/tslib/bin/python。runner会核验主机地址、目录、解释器、nvidia-smi和CUDA；没有本机或CPU运行回退。代码和文档编辑仍通过X盘映射入口。

在远程登录会话中，先确认 hostname、项目目录、nvidia-smi、Python路径和 torch.cuda.is_available()，然后执行下面的命令。CUDA_VISIBLE_DEVICES显式选择远程GPU，先查看是否已有用户训练占用。

## 研究筛选
以下每条命令都需要一个新或空输出目录；完成和部分运行都不能覆盖。
```bash
cd /home/ouyangguanghong/projects/TEST/GTR-main
CUDA_VISIBLE_DEVICES=0 /home/ouyangguanghong/miniconda3/envs/tslib/bin/python -u scripts/gtr_guided_adafe/run_pilot.py --dataset PEMS04 --output reports/gtr_guided_adafe_next/pems04
CUDA_VISIBLE_DEVICES=0 /home/ouyangguanghong/miniconda3/envs/tslib/bin/python -u scripts/gtr_guided_adafe/run_pilot.py --dataset PEMS07 --output reports/gtr_guided_adafe_next/pems07
```

默认五个arms：GTR-only/RevIN1，修复AdaFE/RevIN1，修复AdaFE/RevIN0，新归一化逐token scorer/无guide/RevIN1，同一新scorer/有guide/RevIN1。共有参数初始权重、窗口索引、shuffle和forward RNG一致并写入哈希。固定主学习率3e-4、scorer倍率0.1、clip norm1；1024训练窗口、256验证窗口、预测24步、6轮、batch16、width512。只看验证集选择最佳轮，完全不访问测试集。

可以用 --arms 选择上述一个或多个arm。原型类本身支持RevIN0，但当前五组声明把有/无guide比较固定在RevIN1；如果RevIN0优胜，下一轮应该在RevIN0下配对比较两个guide开关，不能从当前比较外推。

## 完整训练/验证
--full 使用完整训练与验证窗口，例如：
```bash
CUDA_VISIBLE_DEVICES=0 /home/ouyangguanghong/miniconda3/envs/tslib/bin/python -u scripts/gtr_guided_adafe/run_pilot.py --dataset PEMS07 --full --epochs 30 --pred-len 24 --seed 2026 --output reports/gtr_guided_adafe_full/pems07_24_2026
```
这仍是研究训练与验证；正式成绩需要固定结构与超参后再进行一次测试，并扩展到三种子与各预测长度。

## 权重诊断与契约
diagnose_checkpoints.py解析现有日志并严格加载对应权重，探测固定权重下的batch上下文、分支关闭和梯度。它不会更新权重。旧生产日志/权重名没有编码RevIN，正式重跑可能覆盖；调用前必须核对实际日志配置与checkpoint版本，不应直接拿旧日志解释新权重。
```bash
CUDA_VISIBLE_DEVICES=0 /home/ouyangguanghong/miniconda3/envs/tslib/bin/python -m unittest discover -s tests -p test_gtr_guided_adafe.py -v
```
2026-10-13远程8项通过。证据见 reports/gtr_guided_adafe_20261013/contracts.log，完整诊断见 docs/gtr-guided-adafe-diagnosis-20261013.md。


## RevIN关闭后的周期指导配对
run_norm0_pilot.py复用同一训练流程，在PEMS04/07上只训练两个同结构分支：guided_without_prior_norm0和guided_with_prior_norm0。两组均关闭RevIN，只有周期条件开关不同；默认预算、初始化、数据窗口和优化器与前一轮相同。
```bash
CUDA_VISIBLE_DEVICES=0 /home/ouyangguanghong/miniconda3/envs/tslib/bin/python -u scripts/gtr_guided_adafe/run_norm0_pilot.py --dataset PEMS04 --output reports/gtr_guided_adafe_next/norm0_pems04
CUDA_VISIBLE_DEVICES=0 /home/ouyangguanghong/miniconda3/envs/tslib/bin/python -u scripts/gtr_guided_adafe/run_norm0_pilot.py --dataset PEMS07 --output reports/gtr_guided_adafe_next/norm0_pems07
```
2026-10-13本轮输出为reports/gtr_guided_adafe_20261013/norm0_paired_pems04和norm0_paired_pems07，使用新目录避免覆盖之前的RevIN1筛选。研究指标为训练集StandardScaler空间的验证MSE/MAE，没有访问测试集。

