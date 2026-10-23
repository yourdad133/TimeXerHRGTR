# Electricity 96→96 Guide 检查入口

当前状态：脚本已准备，尚未执行。SSH 密钥认证返回 Permission denied (publickey,password)；未加载权重、未产生诊断数值。遵照项目 AGENTS.md，未使用本机 Python、CPU 或 GPU 运行评估。

## 已核对的输入

- 实验日志：experiment_logs/Guide_Electricity/Electricity_96_2026Guide_dff1024.log
- 模型：models.TimeXerGTRGuidedAdaFE.Model
- 任务：Electricity，96→96，seed=2026，dropout=0.05，d_model=512，d_ff=1024。
- 日志最佳验证轮次：19；验证 MSE：0.1129373；训练历史完整至第27轮。
- 检查点：由日志的 start training setting 定位现有 checkpoint.pth。运行时严格加载并记录 SHA-256，复现验证 MSE 后才能认定与本轮匹配。

## 在已有 SSH 登录会话中运行

bash /home/ouyangguanghong/projects/TEST/GTR-main/scripts/gtr_guided_adafe/run_electricity96_probe.sh

入口固定使用远程 tslib 与 GPU 0，复用项目环境核验和显存上限。未启动新训练，不覆盖模型、原日志、result.txt 或已有报告。报告写入新的 reports/electricity96_guide_probe_时间戳 目录。

## 三项检查

1. 固定同一权重、完整验证集和相同输入，比较 Guide 开关前后的硬掩码。阈值均为每个 token 在特征维上的均值。统计掩码元素变化比例、含变化的 token 比例及两种模式的激活比例。
2. 直接测量 adapter 输出减去输入所得到的 AdaFE 残差。报告全局 RMS(残差)/RMS(原 patch)，并分别报告 Guide 开启、关闭，以及开关导致的残差变化。它不是 gamma 数值本身，也不是因果贡献百分比。
3. 保持主路径 GTR 和周期索引不变，只关闭 Guide 的评分条件。按全部目标元素加权计算验证 MSE/MAE，以及两组预测的差值 RMS、MAE 和最大绝对差；不使用测试集选模。

另记录去均值后的 Guide 评分与本地评分的 RMS 比值。按均值阈值选特征时，所有维度相同的评分偏移会抵消，因此不以未去均值的幅值判断指导作用。

## 核验与解释边界

- 固定使用完整验证窗口，保留末尾批次；batch_size 与日志一致。
- 首批检查恢复 Guide 后能否恢复预测；逐批确认主路径 patch 输入和目标保持相同。
- 确认内存参数、检查点文件、数据文件和所记录源代码在检查中保持不变。
- 开启 Guide 的验证 MSE 必须与日志一致：预先设定 rtol=1e-5、atol=1e-6。未通过时标记未核验并以非零状态退出。
- 当前固定权重的干预不能替代从头训练的消融；重叠验证窗口不当作独立样本进行显著性检验。

## 输出

- report.md：三项检查的简明数值报告。
- results.json：完整指标、核验状态和门控值。
- configuration.json、runtime.json、training_log_snapshot.log：配置、环境、哈希和输入日志快照。
- per_window.csv、per_channel.csv：逐窗口、逐变量的验证误差与差值。

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-21
- Verification Status: UNVERIFIED
- Version Label: electricity96_probe_prepared_v1
