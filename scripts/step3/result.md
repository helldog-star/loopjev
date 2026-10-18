# Step 3 结果分析：均匀多轮监督

核对日期：2026-10-18。以下是 seed=0 的 choice 开发实验，非最终测试或多种子结论。
train/dev-cal/validation = 4096/512/2048；validation-check 为1019题（1019个group），
其中979题硬标签用于准确率，全部题用于 NLL/Brier。温度仅在 dev-cal 拟合。
Accuracy 越高越好；NLL（负对数似然损失）和 Brier（概率误差）越低越好。
raw 为原始分数概率，scaled 为温度校准后概率；正温度不会改变准确率。

状态：uniform4 的训练、评测及两项配对比较已完成。

## 共享决策头逐轮指标

- 第1轮：Accuracy 46.48%；NLL raw/scaled = 1.1460/1.1499；Brier raw/scaled = 0.6080/0.6091。
- 第2轮：Accuracy 58.73%；NLL raw/scaled = 0.9536/0.9661；Brier raw/scaled = 0.5126/0.5158。
- 第3轮：Accuracy 62.51%；NLL raw/scaled = 0.8873/0.8835；Brier raw/scaled = 0.4827/0.4817。
- 第4轮：Accuracy 66.50%；NLL raw/scaled = 0.8043/0.8214；Brier raw/scaled = 0.4411/0.4454。

## 本次正式开发结果（seed=0）

两份配对比较已成功补跑：`results/vs-e1-four-loop.json` 和
`results/vs-e1-two-loop.json`（相对 `artifacts/step3/uniform4/`）。
使用 validation-check 1019 题，Accuracy 仅统计其中硬标签题；
以下差值均为 Step3 减 Step2，区间为 1000 次按组 bootstrap 的双侧95%区间。

- Step3 第2轮 vs Step2 四轮模型末轮：Accuracy −3.58 个百分点
  （95%区间 −6.62 到 −0.31），raw NLL +0.0488，scaled NLL +0.0775。
  两种校准口径均未通过预设非劣标准，不能宣称两轮已达到四轮质量。
- Step3 第4轮 vs Step2 四轮模型末轮：Accuracy +4.19 个百分点
  （95%区间 +1.22 到 +7.14），raw NLL −0.1005，scaled NLL −0.0672。
  多轮监督在当前实验中没有损伤末轮，反而呈改善信号。
- Step3 第2轮 vs 独立训练的 Step2 两轮模型末轮：Accuracy +5.31 个百分点
  （95%区间 +2.45 到 +8.39），raw NLL −0.1523，scaled NLL −0.0688。
  说明浅轮质量改善，但仍不能替代上面的四轮质量门槛。

正温度缩放不改变 Accuracy；NLL 越低越好。这里只是单种子开发结果，
未做多重比较校正，不能当作最终测试结论；读取第二轮也不代表已测得提前退出提速。
完整置信区间、任务分组和独立 probe 结果以比较 JSON 为准。


## 如何判断与继续

预设质量非劣要求：Accuracy差值的单侧95%下界 > −1个百分点，
且NLL差值的单侧95%上界 < +0.02；两项必须同时成立。
当前支持“多轮监督改善浅轮读出，并改善末轮质量”，不支持“两轮已替代四轮”。
可以继续Step4对照；若要形成研究结论，应增加训练种子复核。
Step5是否有真实提速收益，还需执行能提前停止的推理实现并实测。

## 证据

- [正式指标](../../artifacts/step3/uniform4/results/metrics.json)
- [训练协议](../../artifacts/step3/uniform4/results/protocol.json)
- [相对四轮基线](../../artifacts/step3/uniform4/results/vs-e1-four-loop.json)
- [相对两轮基线](../../artifacts/step3/uniform4/results/vs-e1-two-loop.json)

独立probe是诊断工具，主结论使用共享头 `multi_loop_head`。
