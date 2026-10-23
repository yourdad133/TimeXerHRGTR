# AdaFE 静态对照资料

这些文件原存于 reports/adafe_v1，2026-10-12 清理生成报告时迁移到本目录。保留上游源码及来源、三份冻结源码和六条匹配命令样本，用于已有回归检查；输出报告仍写入 reports。

stage_b_commands.json 是已记录的固定样本。重新生成实际运行命令使用 scripts/adafe_v1/experiments.py 的 --prepare。已移除的历史模型参考由 snapshot_source 从已有 Git 备份读取。
