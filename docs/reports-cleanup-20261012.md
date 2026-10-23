# reports 存储清理

2026-10-12 按用户要求删除生成的 reports 目录。清理前共有 777 个文件，合计 4,121,074,459 字节；其中 77 个 .pth 权重占 4,112,652,134 字节。

692 个实验配置、指标、逐轮记录、诊断结论、图表及源码原件已逐文件校验 SHA256，归档至 `X:/TEST/maintenance_archives/20261012_reports_cleanup/records/`。归档原件总计 8,399,323 字节，清单为同级 `manifest.json`。服务器位置为 `/home/ouyangguanghong/projects/TEST/maintenance_archives/20261012_reports_cleanup/`。

清理前 Git 快照标签：`before-reports-cleanup-20261012`，提交 `fbb10e790dcde4bd87e7410e253ee7d38e556404`。该快照保存当前代码、当时的结果索引及已跟踪证据；被忽略的权重不在 Git 中，删除后再次评估这些权重需要重训。清理只删除 reports，根目录 result.txt、checkpoints、experiment_logs 和数据集继续保留。

PEMS08 完整对照的配置和结果见归档中的 `adafe_fixes/pems08_full_20261011/` 与 `adafe_fixes/pems08_fixed_slow_horizons_20261012/`；修复结论见 `adafe_fixes/20261011/fix_report.md`，最初诊断见 `adafe_diagnostics/20261011_173843/diagnosis.md`。

测试所需的三份静态源码快照、上游参考和固定命令样本迁移到 `scripts/adafe_v1/reference/`；旧 Electricity 权重探查清单迁移到 `scripts/adafe_fixes/reference/historical_snapshot.json`。正式模型计算和训练配置保持原有行为。未来运行专用诊断脚本时，它们仍会生成新的 reports 输出。

删除 reports 后，在 SSH / tslib / GPU 1 上完成 12 项 AdaFE 回归检查，全部通过；四份改动 Python 文件通过语法检查，Git 差异检查通过。测试日志位于归档根目录的 adafe_reference_tests.log。
