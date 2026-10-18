# LoopJev 实验脚本导航

每个阶段分开维护使用说明与结果分析；所有实验命令经 `python scripts/run.py` 启动。

## 阶段入口

- Step0 模型接口：[使用说明](step0/README.md) · [检查结果](step0/result.md)
- Step1 冻结表示：[使用说明](step1/README.md) · [结果分析](step1/result.md)
- Step2 末轮监督：[使用说明](step2/README.md) · [结果分析](step2/result.md)
- Step3 多轮监督：[使用说明](step3/README.md) · [结果分析](step3/result.md)
- Step4 稠密及外部对照（正式任务已启动，结果待完成）：[使用说明](step4/README.md) · [结果分析与待完成项](step4/result.md)
- Step5 提前退出：[设计说明](step5/README.md) · [实验状态](step5/result.md)
- [环境与下载](setup/README.md) · [公共工具](common/README.md)

## 文档与产物

README写用途、配置和运行；result.md写结果与证据；PROTOCOL.md写固定实验方法。
规则唯一正文为根目录 [AGENTS.md](../AGENTS.md)，Cursor规则文件只提供加载入口。

代码放阶段目录；日志放 `logs/`；指标、协议、预测和模型增量放 `artifacts/.../results/`；
可复用特征放 `cache/`；临时文件放 `work/`；预训练模型和数据放 `models/`、`data/`。
已有实验不覆盖；复现用新目录，比较用新输出文件。

## 统一运行入口

```bash
cd /root/siton-tmp/home/liuxinyu/loopjev
PY=/root/siton-tmp/home/liuxinyu/miniconda3/envs/loopjev/bin/python
$PY scripts/run.py --list
$PY scripts/run.py step3 train --help
```

统一入口设置工作目录和跨阶段导入路径；不要直接执行阶段Python文件或恢复根目录旧脚本链接。
`common/runtime_paths.py`定位历史指纹文件名对应的实际源码，仍校验真实字节哈希。
`--help`用于Python实验命令；不要运行环境安装或矩阵任务来试探帮助。

Open-Jev官方源码统一保存在 `repos/Open-Jev/`，固定提交与环境要求见Step4使用说明。
