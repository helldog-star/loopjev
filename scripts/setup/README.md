# 环境与资源准备

## 做什么

这里负责准备运行环境和下载资源，不负责训练模型，也没有模型效果结论。

- `setup_env.sh`：创建 loopjev conda 环境并安装依赖。
- `download_assets.py`：从配置的 Hugging Face 镜像下载资源。

## 当前状态与注意事项

服务器已有loopjev主环境和openjev独立环境；用途和重建方法见
[环境文档](../../.envs/README.md)。
不要在训练过程中重建环境或升级依赖。
现有实验协议记录 torch 2.8.0+cu128、transformers 4.55.0、peft 0.21.2。
安装脚本并未锁死所有包版本；复现实验应对照.envs/loopjev-requirements.txt及对应protocol.json。

下载脚本当前启用的任务只有 `openjev2b`，Ouro 与数据集任务在源码中被注释。
不要以为执行它会自动下载所有项目资源。默认 HF_ENDPOINT 为 hf-mirror.com，
缓存放项目 `.hf_cache/`；默认资源根目录写在脚本 ROOT 中。

## 需要时如何运行

以下是实际安装/下载命令，不是检查命令；已有资源时不要重复执行。

```bash
cd /root/siton-tmp/home/liuxinyu/loopjev
PY=/root/siton-tmp/home/liuxinyu/miniconda3/envs/loopjev/bin/python
# 仅首次准备新环境，并且 conda 可用时：
bash scripts/setup/setup_env.sh
# 下载配置中已经启用的 Open-Jev 外部基线：
$PY scripts/run.py setup download openjev2b
```

模型放 models/，数据放 data/；不要把资源文件放入 scripts/。
