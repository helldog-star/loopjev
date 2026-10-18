# LoopJev

研究循环模型的决策读出、多轮监督以及质量与计算成本的关系。
研究设想见 [idea.md](idea.md)，计划见 [plan.md](plan.md)，协作规范见 [AGENTS.md](AGENTS.md)。

## 先选择正确的环境

项目的**主环境是loopjev**，用于Ouro实验、Qwen稠密基线及结果比较。
Open-Jev是单独加入的外部基线，只有它的模型推理使用独立openjev环境。
两套环境的Transformers等依赖不同，不能互相覆盖升级。

```bash
cd /root/siton-tmp/home/liuxinyu/loopjev
PY=/root/siton-tmp/home/liuxinyu/miniconda3/envs/loopjev/bin/python
JEVPY="$PWD/.envs/openjev/bin/python"
$PY scripts/run.py --list
# Ouro/Qwen使用主环境：
$PY scripts/run.py step4 train --help
# Open-Jev模型评测使用独立环境：
$JEVPY scripts/run.py step4 external-eval --help
```

两套环境的版本、依赖及重建方法集中在 [.envs/README.md](.envs/README.md)。
主环境本体位于项目外的miniconda3/envs/loopjev；.envs保存其清单，不复制或迁移环境。
所有CPU配对比较均可用主环境运行。

## 实验与源码入口

- [阶段导航](scripts/README.md)：各Step的README负责使用说明，result.md负责结果分析。
- [外部源码版本](repos/README.md)：恢复Open-Jev和jevk5的固定提交。
- [环境文档](.envs/README.md)：主环境与外部基线环境的安装、选择和验证。

运行前先读阶段README，使用新的输出目录，不覆盖已有结果。模型、数据及原始结果需另行准备；
仅克隆代码仓库不能恢复服务器上的实验大文件。

## Git保存范围

提交代码、研究与结果摘要、外部源码版本及.envs下的依赖清单和说明。
不提交第三方仓库实体、完整环境、models/data/artifacts/logs/work及缓存。
文档中的原始结果链接在仅克隆代码时可能不存在；权重和实验产物需要单独备份。
