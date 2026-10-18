# Step 1 结果分析：冻结表示的决策信息

核对日期：2026-10-18。以下是 seed=0 的 choice 开发实验，非最终测试或多种子结论。
train/dev-cal/validation = 4096/512/2048；validation-check 为1019题（1019个group），
其中979题硬标签用于准确率，全部题用于 NLL/Brier。温度仅在 dev-cal 拟合。
Accuracy 越高越好；NLL（负对数似然损失）和 Brier（概率误差）越低越好。
raw 为原始分数概率，scaled 为温度校准后概率；正温度不会改变准确率。

## 逐轮独立读出器

- 第1轮：Accuracy 38.30%；NLL raw/scaled = 1.3376/1.3346；Brier raw/scaled = 0.6800/0.6782。
- 第2轮：Accuracy 43.21%；NLL raw/scaled = 1.2413/1.2502；Brier raw/scaled = 0.6337/0.6373。
- 第3轮：Accuracy 49.54%；NLL raw/scaled = 1.1447/1.1559；Brier raw/scaled = 0.5900/0.5939。
- 第4轮：Accuracy 48.01%；NLL raw/scaled = 1.1618/1.1769；Brier raw/scaled = 0.6000/0.6055。

## 已有结果（2026-10-18 核对）

依据 `artifacts/step1_expanded/results/protocol.json`、`metrics.json` 与
`logs/step1_finish.log`。选中 300 步、L2=0.01。
使用 4096 行训练时，check 上第 1–4 轮准确率分别是：
**38.30%、43.21%、49.54%、48.01%**。
对应 raw NLL 为 **1.3376、1.2413、1.1447、1.1618**。

第 3 轮比第 4 轮准确率高 1.53 个百分点，但配对 95% 区间约为 −1.02～+4.15；
NLL 差约 −0.0172，区间约 −0.0366～+0.0025，均包含 0。
第 4 轮温度为 0.8818，scaled NLL 为 1.1769，反而略差于 raw。

结论：较深表示具有可读出的决策信息，值得进入 LoRA 基线；
不能宣称第 4 轮必然最优，也不能认定第 3 轮显著优于第 4 轮。


## 限制与下一步

这是冻结骨干、单种子、choice 开发结果。check 不是独立最终测试集；
检索/匹配控制组样本很少，不能可靠检验“推理任务比查表任务更受益”。
Step 2 将更新 LoRA，届时必须重提特征，不能沿用冻结骨干的 hidden states。


## 证据

- [指标与配对区间](../../artifacts/step1_expanded/results/metrics.json)
- [固定协议](../../artifacts/step1_expanded/results/protocol.json)

独立 probe 指冻结骨干后训练的线性读出器。这里的逐轮差异不能替代多轮监督的消融实验。
