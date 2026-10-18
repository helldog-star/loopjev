# Step 4 结果分析：稠密与外部模型对照

## 当前完成范围

Qwen的300步训练及正式评测、Open-Jev真实权重冒烟及完整外部评测全部完成。
Qwen四项和Open-Jev五项配对比较均成功，主日志确认 `Step4 pipeline completed`。
本阶段已具备单种子开发结论；不等于多种子或独立最终测试结论。

## 评测口径

seed=0，train/dev-cal/validation=4096/512/2048；validation-check为1019题，
其中979题硬标签用于准确率，全部1019题用于NLL/Brier。
下游训练为300次更新×16题，固定候选顺序及样本调度；不同模型并非等token或等算力。
raw是原始概率，scaled是仅在dev-cal拟合温度后的概率；正温度不改变准确率。
Accuracy越高越好；NLL（负对数似然损失）、Brier（概率误差）越低越好。

## Qwen主模型表现

- validation-check：Accuracy **49.13%**；NLL raw/scaled =
  **1.3551/1.1695**；Brier raw/scaled =
  0.6293/0.5922。
- 完整validation（2048题，其中1957题硬标签）：Accuracy 50.03%；
  NLL raw/scaled = 1.3342/1.1569。
- 温度校准改善了概率损失，但不会修正错误候选的排序。

## 与Ouro的配对比较

以下差值方向均为Qwen减Ouro，Accuracy差使用百分点；区间为1000次group bootstrap双侧95%区间。
主判据是预设的单侧非劣界限：Accuracy下界 > −1个百分点且NLL上界 < +0.02。

### 相对E1四轮末轮

Ouro准确率62.31%；Qwen差值 **-13.18个百分点**
（95%区间-16.77～-9.74）。
NLL差 raw/scaled = +0.4503/+0.2809；
raw NLL的95%区间为+0.3485～+0.5455，
scaled为+0.2075～+0.3509。
两种口径均未通过Qwen相对参考的预设非劣标准。

### 相对E1独立两轮末轮

Ouro准确率53.42%；Qwen差值 **-4.29个百分点**
（95%区间-7.51～-1.02）。
NLL差 raw/scaled = +0.2492/+0.1346；
raw NLL的95%区间为+0.1714～+0.3297，
scaled为+0.0819～+0.1870。
两种口径均未通过Qwen相对参考的预设非劣标准。

### 相对E2四轮模型的第二轮读出

Ouro准确率58.73%；Qwen差值 **-9.60个百分点**
（95%区间-13.18～-6.10）。
NLL差 raw/scaled = +0.4015/+0.2034；
raw NLL的95%区间为+0.3049～+0.4955，
scaled为+0.1323～+0.2698。
两种口径均未通过Qwen相对参考的预设非劣标准。

### 相对E2四轮末轮

Ouro准确率66.50%；Qwen差值 **-17.36个百分点**
（95%区间-21.04～-13.84）。
NLL差 raw/scaled = +0.5508/+0.3481；
raw NLL的95%区间为+0.4533～+0.6484，
scaled为+0.2739～+0.4206。
两种口径均未通过Qwen相对参考的预设非劣标准。

## 成本与取舍

Qwen实测推理约 **60.98 ms/题**，训练本体约
28.19分钟，不含特征重提、probe和评测。
计时采用固定32题、1次预热、3次重复，范围是预分词输入的GPU前向与决策头，不是完整业务延迟。

- E1独立两轮末轮：109.50 ms/题；Qwen耗时约为其55.7%。
- E1四轮末轮：216.29 ms/题；Qwen耗时约为其28.2%。
- E2四轮末轮：217.04 ms/题；Qwen耗时约为其28.1%。

因此当前观察到质量与延迟的取舍，不能宣称Ouro在质量和成本上全面优于稠密模型。
E2第二轮分数仍来自实际执行四轮的模型，不代表已实现两轮退出；跨模型速度不等同FLOPs比。

## 可以和不能得出的结论

当前配置下，Ouro E1/E2的决策质量优于这一个Qwen稠密基线，Qwen则更快。
这支持继续研究循环模型在给定预算下的质量—成本关系，不足以证明循环结构的因果优势。
模型预训练、tokenizer、提示包装、初始化尺度及总计算量仍不同；同一优化配置也未必对两模型同样合适。

Qwen独立probe准确率为53.12%，高于主决策头49.13%；
这是读出/优化仍可能改善的信号，不能将当前主头成绩视作Qwen能力上限。

这些是单种子开发结果；bootstrap不包含训练种子波动，未做多重比较校正，check不是最终测试集。
小任务组（reasoning44题、retrieval/matching9题）不能支持强任务机制结论。

## Open-Jev外部基线结果

固定发布checkpoint完整评测已完成：dev-cal512题、validation2048题，未重新训练权重。
下列主指标使用validation-check1019题，其中979题硬标签用于Accuracy。

- 原始概率：Accuracy 48.31%；NLL 1.6646；Brier 0.7234。
- 发布温度：Accuracy 48.31%；NLL 1.3374；Brier 0.6553。
- 本项目dev-cal温度：Accuracy 48.31%；NLL 1.2089；Brier 0.6114。

发布温度为1.5188，dev-cal重新拟合温度为2.1639。
完整validation准确率48.75%；其raw/dev-cal NLL为1.6314/1.1990。
温度校准改善概率损失，但不会改变候选排序或准确率。

### 配对差异

方向为Open-Jev减参考模型；Accuracy差值为百分点，区间为1000次group bootstrap双侧95%区间。

- 相对E1四轮：Accuracy -13.99（95%区间-18.28～-10.02）；
  双方dev-cal校准后的NLL差+0.3203（95%区间+0.2444～+0.3944）。
- 相对E1独立两轮：Accuracy -5.11（95%区间-9.39～-0.93）；
  双方dev-cal校准后的NLL差+0.1740（95%区间+0.1092～+0.2449）。
- 相对E2第二轮读出：Accuracy -10.42（95%区间-14.51～-6.37）；
  双方dev-cal校准后的NLL差+0.2428（95%区间+0.1695～+0.3151）。
- 相对E2第四轮：Accuracy -18.18（95%区间-22.23～-14.23）；
  双方dev-cal校准后的NLL差+0.3875（95%区间+0.3089～+0.4569）。
- 相对Qwen稠密基线：Accuracy -0.82（95%区间-5.01～+3.65）；
  双方dev-cal校准后的NLL差+0.0394（95%区间-0.0336～+0.1155）。

当前Open-Jev相对四个Ouro参考的准确率和校准后NLL均更差，相关区间未跨0。
与Qwen相比，准确率及双方dev-cal校准后的NLL区间都跨0，不能据点估计排出可靠高低，
也不能把“没有检测到差异”说成两者等效。raw NLL下Open-Jev更差，校准后差距明显缩小。
发布温度与参考raw的比较是不同校准条件，不混作双方同口径的胜负证据。

### 外部结果的解释边界

Open-Jev使用后训练的Qwen3.5-2B，历史训练量、数据和目标与本项目不同，潜在数据重合未排除。
这只是固定checkpoint在本项目choice开发划分上的表现，不代表官方整体能力，也不能推出后训练有害。
其推理使用PyTorch回退实现，缺少可选快速算子；计时包含分词、padding和传输。
因此不与Ouro/Qwen的预分词计时直接比较速度，不用该结果判断架构效率。

## Step4阶段结论与下一步

本次计划的Qwen受控下游基线和Open-Jev外部基线均已完成，所有九份跨模型比较已生成。
可以形成单种子开发结论：当前配置下，Ouro多轮监督模型在所测质量指标上表现最好；
Qwen质量较低但推理更快；Open-Jev没有在当前划分上优于Ouro，且不能可靠区分其与Qwen的准确率。

不能把这些结果上升为“循环结构因果上优于稠密结构”或“质量与算力全面占优”。
主实验仍需多种子复核、检查Qwen读出和优化是否充分，并用未参与开发的测试划分做最终验证。
下一阶段可进入Step5的离线退出策略分析，但实际提速必须实现提前停止并重新计时；
Step3的第二轮仍未追平四轮E1，不能预设固定两轮退出已经满足质量要求。

外部结果证据：[metrics.json](../../artifacts/step4/openjev2b/results/metrics.json)、
[protocol.json](../../artifacts/step4/openjev2b/results/protocol.json)。

- [E1四轮配对比较](../../artifacts/step4/openjev2b/results/vs-e1-four-loop.json)
- [E1独立两轮配对比较](../../artifacts/step4/openjev2b/results/vs-e1-two-loop.json)
- [E2第二轮读出配对比较](../../artifacts/step4/openjev2b/results/vs-e2-loop2.json)
- [E2第四轮配对比较](../../artifacts/step4/openjev2b/results/vs-e2-loop4.json)
- [Qwen稠密基线配对比较](../../artifacts/step4/openjev2b/results/vs-qwen.json)

## 证据入口

- [正式指标](../../artifacts/step4/qwen3_1p7b_qv_r8/results/metrics.json)
- [实际训练协议](../../artifacts/step4/qwen3_1p7b_qv_r8/results/protocol.json)
- [E1四轮末轮比较](../../artifacts/step4/qwen3_1p7b_qv_r8/results/vs-e1-four-loop.json)
- [E1独立两轮末轮比较](../../artifacts/step4/qwen3_1p7b_qv_r8/results/vs-e1-two-loop.json)
- [E2四轮模型的第二轮读出比较](../../artifacts/step4/qwen3_1p7b_qv_r8/results/vs-e2-loop2.json)
- [E2四轮末轮比较](../../artifacts/step4/qwen3_1p7b_qv_r8/results/vs-e2-loop4.json)
