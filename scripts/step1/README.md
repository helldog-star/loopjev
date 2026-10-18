# Step 1：冻结 Ouro，看看各轮表示能读出多少决策信息

使用方法见本页；实验结果、解释及限制见 [result.md](result.md)。

## 要回答什么

先不更新 Ouro 骨干，只训练小型线性读出器（probe）。如果较深的隐藏表示可以被更好地读出，
说明后续训练决策模型值得尝试；但不能直接说明多轮监督或提前退出有效。


## 脚本分工

- `step1_extract.py`：固定样本与候选顺序，提取各轮最后 token 的隐藏态，保存缓存。
- `step1_probe.py`：训练、评估逐轮独立线性 probe；保留作为基础工具和小规模诊断入口。
- `step1_finish.py`：当前 expanded 实验的主分析入口，按事先固定的协议选择配置、
  计算样本规模曲线、分组指标及配对置信区间。
- 通用函数位于 `../common/step1_common.py`，由统一入口配置模块路径后复用。


## 已锁定的实验设置

choice 数据：train/dev-cal/validation = 4096/512/2048，每 group 一行。
validation 根据固定 group 哈希分成 select=1029 和 check=1019；
select 选择统一配置，check 用于开发期复核，温度只在 train 派生的 dev-cal 上拟合。

比较 steps={100,300} 与 L2={0.001,0.01}，学习率 0.01，损失 CE+0.1 Brier。
用 select 上四轮平均 raw NLL 最低的配置统一评估所有轮次。
样本规模曲线使用同一训练序列的 512/2048/4096 前缀。


## 如何复现

现有产物在 `artifacts/step1_expanded/`，不要覆盖。
下例只用于确实需要独立复现时，使用新的目录；当前下游训练无需重提冻结特征。

```bash
cd /root/siton-tmp/home/liuxinyu/loopjev
PY=/root/siton-tmp/home/liuxinyu/miniconda3/envs/loopjev/bin/python
OUT=artifacts/step1_reproduce
$PY scripts/run.py step1 finish --out-dir "$OUT/results" --write-protocol
$PY scripts/run.py step1 extract --model models/Ouro-1.4B --data data/Open-Jev-v1.1   --out-dir "$OUT/cache" --rows train=4096,dev-cal=512,validation=2048   --seed 0 --max-length 4096 --batch-size 8 --token-budget 4096 --verify-batching
$PY scripts/run.py step1 finish --cache-dir "$OUT/cache" --out-dir "$OUT/results"
```


## 输入与产物

输入：Ouro与Open-Jev数据；`cache/`保存逐轮特征，`results/`保存协议、probe指标和比较。
当前实验根目录为 `artifacts/step1_expanded/`。
