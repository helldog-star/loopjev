# Step 2：建立只监督最后一轮的决策基线

使用方法见本页；实验结果、解释及限制见 [result.md](result.md)。

## 实验目的

把 Ouro 的隐藏表示变成每个候选答案的分数，再在同一道题的候选之间做 softmax。
本步只监督最后一轮，用来确认模型能否学会更好的决策，并给 Step 3 提供公平对照。


## 脚本分工

- `step2_train.py`：运行一个条件，训练、重新提取必要特征、评估、保存最终产物。
- `step2_evaluate.py`：逐轮共享头与独立 probe 评估、温度拟合和 group bootstrap。
- `step2_compare.py`：在已完成条件之间做配对比较，不会重新训练。
- `PROTOCOL.md`：保留的英文技术协议，记录更细的实现、成本定义和原始验证信息。


## 四个条件

- head4：骨干冻结，只训练四轮末轮的共享决策头。
- head2：模型实际只循环两轮，只训练第二轮决策头。
- lora4：原始 Ouro + 全新 head + LoRA r8，只监督第四轮。
- lora2：原始 Ouro + 全新 head + LoRA r8，只监督第二轮。

四组独立初始化，不互相接续训练；数据顺序与 seed 相同。
300 次参数更新，每次 16 道题；head LR=.001，LoRA LR=.0002；
CE+0.1 Brier，raw-head L2=.001，梯度裁剪 1，LoRA 仅更新 q_proj/v_proj。
默认共4800次题目呈现；相同步数不代表相同计算量。


## 运行方式

以下训练示例使用新目录；现有正式结果在 `artifacts/step2/{head4,head2,lora4,lora2}`。
切换 --mode 和 --loops 可运行其他条件，不要并行启动 GPU 训练。

```bash
cd /root/siton-tmp/home/liuxinyu/loopjev
PY=/root/siton-tmp/home/liuxinyu/miniconda3/envs/loopjev/bin/python
$PY scripts/run.py step2 train --mode lora --loops 4   --model models/Ouro-1.4B --data data/Open-Jev-v1.1   --cache-dir artifacts/step1_expanded/cache --out-dir artifacts/step2/lora4_reproduce
# 对现有结果做新的比较输出，指定尚不存在的文件：
$PY scripts/run.py step2 compare --candidate artifacts/step2/lora2   --reference artifacts/step2/lora4 --out artifacts/step2/comparisons/lora2-lora4_recheck.json
```


## 输入与产物

输入为原始Ouro、数据与Step1固定缓存。各条件 `results/` 保存协议、head、adapter（LoRA）、
probe、预测和指标；`cache/`保存更新后特征。运行日志为 `logs/step2_matrix.log`。
