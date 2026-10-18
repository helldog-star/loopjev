# Step 0：先确认模型接口能正确使用

使用方法见本页；实验结果、解释及限制见 [result.md](result.md)。

## 实验目的

在训练前确认 Ouro 可以输出每次循环的隐藏表示和门值，确定如何从这些表示读出分数。
这一步是程序和数据接口检查，不是在评测研究方法是否有效。


## 脚本与运行方法

`step0_smoke.py` 检查 Yes/No 分词、循环输出形状、末轮隐藏态与完整模型 logits 的一致性。
提供 --data 时还检查数据文件结构并打印少量训练样例。

```bash
cd /root/siton-tmp/home/liuxinyu/loopjev
PY=/root/siton-tmp/home/liuxinyu/miniconda3/envs/loopjev/bin/python
$PY scripts/run.py step0 smoke --model models/Ouro-1.4B --data data/Open-Jev-v1.1
```

不要为了查看说明重新占用训练中的 GPU；参数帮助可用 `step0 smoke --help`。


## 输入与产物

输入为Ouro模型与可选数据集；诊断输出写入终端/日志，不产生模型checkpoint。
