# LoopJev

研究循环模型的决策读出、多轮监督以及质量与计算成本的关系。
研究设想见 [idea.md](idea.md)，实验计划见 [plan.md](plan.md)，协作规范见 [AGENTS.md](AGENTS.md)。

## 实验入口

- [各阶段脚本与使用说明](scripts/README.md)：Step0–5分别维护README与result.md。
- [外部源码固定版本](repos/README.md)：按说明恢复Open-Jev和jevk5。
- [Open-Jev环境重建](.envs/README.md)：依赖清单与Conda YAML。

```bash
python scripts/run.py --list
```

运行实验前先阅读对应阶段README，使用匹配的Python环境。模型和数据需另外准备；
只克隆本仓库不能直接复原服务器上的大文件。已有实验目录不得覆盖。

## 版本控制边界

Git保存项目代码、方法与结果摘要、外部源码版本、依赖及重建说明。
repos下的完整第三方仓库、.envs下的实际环境、models/data/artifacts/logs/work不提交。
文档中的服务器产物链接在仅克隆代码时可能不存在；原始结果和模型需要单独备份。
依赖清单提升可复现性，但不保证跨平台二进制完全一致，也不代替数据与模型版本校验。
