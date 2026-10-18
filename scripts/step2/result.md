# Step 2 结果分析：末轮监督基线

核对日期：2026-10-18。以下是 seed=0 的 choice 开发实验，非最终测试或多种子结论。
train/dev-cal/validation = 4096/512/2048；validation-check 为1019题（1019个group），
其中979题硬标签用于准确率，全部题用于 NLL/Brier。温度仅在 dev-cal 拟合。
Accuracy 越高越好；NLL（负对数似然损失）和 Brier（概率误差）越低越好。
raw 为原始分数概率，scaled 为温度校准后概率；正温度不会改变准确率。

## 四个条件的最终轮结果

### head4

- 第4轮：Accuracy 43.41%；NLL raw/scaled = 1.2379/1.2397；Brier raw/scaled = 0.6501/0.6511。

### head2

- 第2轮：Accuracy 37.79%；NLL raw/scaled = 1.3028/1.3090；Brier raw/scaled = 0.6717/0.6769。

### lora4

- 第4轮：Accuracy 62.31%；NLL raw/scaled = 0.9048/0.8886；Brier raw/scaled = 0.4753/0.4725。

### lora2

- 第2轮：Accuracy 53.42%；NLL raw/scaled = 1.1059/1.0349；Brier raw/scaled = 0.5749/0.5557。



## 已有结果与结论（2026-10-18 核对）

依据各条件 `results/metrics.json`、`protocol.json` 与
`artifacts/step2/comparisons/lora4-head4.json`、`lora2-lora4.json`。
主分析使用 validation-check：1019 行，其中 979 行硬标签用于准确率，全部行用于 NLL/Brier。

1. **四轮 LoRA 有明确收益。** head4 → lora4 的准确率为 43.41% → 62.31%，
  raw NLL 为 1.2379 → 0.9048，Brier 为 0.6501 → 0.4753。
   准确率提升 18.90 个百分点，配对 95% 区间 +15.17～+22.77。
   完整 validation 上准确率为 43.28% → 60.09%，方向一致。
2. **两轮更快但尚未追平四轮。** lora2 准确率 53.42%、raw NLL 1.1059；
  相比 lora4 准确率低 8.89 个百分点，95% 区间 −11.99～−5.70。
   raw/scaled 均未通过预设非劣要求：Accuracy 单侧下界 > −1 个百分点，且 NLL 上界 < +0.02。
3. **深轮决策更好，不代表所有校准指标都单调变好。** lora4 共享头第 1–4 轮准确率为
  32.28%、48.72%、57.81%、62.31%；第 2 轮温度缩放使 NLL 从 1.1352 恶化到 1.2522。
   后续仍应同时报告 raw/scaled，不可只挑有利结果。
4. **head-only 仍可能有优化空间。** head4 准确率低于同骨干独立 probe 的 48.01%；
  训练配置和标准化不同，不能直接归因于表示无效或代码错误。

LoRA 两轮/四轮实测推理延迟约 109.50/216.29 ms/题，约 1.98 倍速度差。
只基于固定 32 题、预分词输入、1 次预热+3 次计时；不包含分词，不能当成 FLOPs 或完整业务延迟。
训练本体约 47.12/95.92 分钟，不含重提特征、probe 和评估。

结论：Step 2 已建立有效基线，支持继续 Step 3；尚未证明浅轮能保持深轮质量。
check 推理组仅 44 行、检索/匹配组仅 9 行，且仅一个训练种子，研究结论仍有限。

## 证据与下一步

各条件指标与协议位于 `artifacts/step2/{head4,head2,lora4,lora2}/results/`。
主要比较：[LoRA相对head-only](../../artifacts/step2/comparisons/lora4-head4.json)、
[两轮相对四轮](../../artifacts/step2/comparisons/lora2-lora4.json)。
配对区间为按group进行1000次bootstrap的双侧95%区间；非劣判定使用预设单侧界限。
Step3从原始预训练模型独立初始化，检验多轮监督，而不是接续本步权重训练。