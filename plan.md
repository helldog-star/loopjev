# LoopJev：循环深度上的决策校准与自适应退出

## 研究问题

在 Ouro-1.4B 的循环隐藏状态上加入 Open-Jev 式 candidate-wise decision head，研究：

1. 各循环深度的决策质量（准确率、概率校准、排序）如何变化？
2. 多轮监督能否让浅轮达到深轮 final-only 的质量？
3. 决策级退出能否在相同平均循环数下优于固定深度和 Ouro 自带的 token 级门？

Ouro 的门是 token 级、按语言模型损失训练；LoopJev 的退出是决策级、按决策风险和校准评估。

## Ouro 何时变成 Jev 形式

“Jev 形式”指把语言模型输出改造成候选决策分布：每个候选渲染成独立 `candidate_prompt`，读取最后 token 隐藏态，经标量头得到候选分数，再在候选维度做 softmax。

- **Step 0：还不是 Jev。** 只验证 Ouro 的循环输出、门、词表和数据接口。
- **Step 1：开始 Jev-style 推理，但不训练 Ouro 骨干。** Zero-shot 不更新参数；probe 会训练小型读出头，作为诊断。
- **Step 2：正式建立 LoopJev 决策基线。** 在 Ouro 上挂 `Linear(hidden_size, 1)` decision head，用决策监督训练；先只训练 head，再加入 LoRA。LoRA 会改变共享循环骨干，因而影响全部循环轮次。
- **Step 3：训练多轮 LoopJev。** 对每轮输出加监督，使浅轮也能直接决策。

所以，“采用 Jev 的打分范式”从 **Step 1** 开始；第一次训练读出参数发生在 **Step 1 的 probe**；正式建立决策模型从 **Step 2 的 E1** 开始；真正更新 Ouro 骨干则从 **Step 2 加入 LoRA** 开始。本工作使用监督式 CE/Brier，不声称复现官方未公开的 RLCD，也不生成自回归 reasoning trajectory。

## 假设与 Go/No-go

- H1：循环深度可能分别影响准确率、概率质量和风险排序，分别用 Accuracy、NLL/Brier/ECE、AURC/Risk–Coverage 检验；报告 raw 和逐轮 temperature-scaled 结果，不要求所有指标同时改善。
- H2：Final-only 监督会使早期轮相对不易读出；用每轮独立 probe 和跨轮共享头两种方式检验。
- H3：深度对推理密集任务更有帮助，对查表类任务帮助较小。
- 第 14 天只用 validation 做 Go/No-go。试验前锁定实用阈值：Accuracy 非劣界 1 个百分点，NLL 非劣界 0.02 nats/row；对 multi-loop 第 2 轮减四轮 final-only 的差值做 group bootstrap，Accuracy 单侧 95% 下界需大于 -0.01，NLL 单侧 95% 上界需小于 0.02。先跑单种子 pilot，有信号后用至少 3 个种子复核。
- No-go：E0/E1/E2 后 validation 推理子集的深度差异仍在阈值内且没有质量—计算收益。置信区间过宽时标记“证据不足”，补样本，不把不显著直接当作无效。

## 资源、数据和隔离

- 模型：`ByteDance/Ouro-1.4B`；`OuroModel.forward` 返回每轮归一化隐藏态 `hidden_states_list` 和 token 级 `gate_list`。
- 数据：`ZefanCai/Open-Jev-v1.1`。用 `record_json` 恢复记录；target、metadata 和控制 latent 不进入模型输入。
- 首轮 pilot 只做 `choice`，固定推理子集及查表/匹配控制子集；验证数据链路后加入 `noul`。`score` 单独定义输出和指标后再加入。补充推理任务先选 ARC-Challenge 和 LogiQA/ReClor 中的一项，独立封存测试集。
- `train` 拟合 probe/head/LoRA；`validation` 选择训练超参、深度和模型。开发阶段的温度诊断使用 train 内按 group 预留的 dev-cal；正式 `calibration` 在方案锁定后才用于最终温度拟合；`test/ood` 只作最终报告。
- 已核验各 split 间没有共享 `group_id`。首轮按 source/kind/group 分层抽样，不能把训练集前两万行当作总体分布。
- 当前决策打分显式关闭 `use_cache`，规避 Ouro 自定义 cache 与 Transformers 4.55 的只读 `key_cache` 冲突；此设置不会减少循环数。锁定实际包版本、模型/数据 revision 与代码哈希。

## 方法约定

- 已按 Open-Jev `bd4118882f733574a3250a4b65fe4d884130c08b` 的 `api.py` 核验 candidate renderer，一致性检查通过；固定官方源码在 `repos/Open-Jev/`。训练代码位于本项目 `scripts/`，JevK5 仅作参考。模型输入仅包含 state/question 和当前候选，使用 Ouro chat template。
- 每个候选独立编码，可批量前向；记录候选顺序、长度和截断率，读取最后一个非-padding token。同一题候选同步置换 options 和 target；choice loss 按 decision row 归一。
- 共享 `Linear(hidden_size, 1)`；choice 在候选维度 softmax。noul 定义 `p_yes=sigmoid(z)`、`p_no=1-p_yes`，target 顺序严格为 `[no, yes]`。
- 头可用 `lm_head` 的 `Yes − No` 行初始化，bias 为 0；先核对真实 prompt 后缀下的 token ID。
- 使用 soft CE + `0.1 × Brier`；多轮权重归一化为 `sum(alpha_t)=1`。保留 CE-only、共享头、每轮独立头、共享头加每轮温度消融。

## 试验步骤

### Step 0：冒烟

断言循环数和门形状；检查每轮 logits、末轮与完整模型 logits 的一致性；检查 Yes/No tokenization；用真实 candidate renderer 检查 prompt、候选顺序、长度和截断。

### Step 1：冻结 Ouro 的逐轮曲线与 E0 probe

在 train 抽样拟合每轮独立线性 probe；对冻结 Ouro 做 candidate-wise zero-shot 读出；缓存每轮最后 token 隐藏态与门 `lambda`。输出 Accuracy、NLL、Brier、ECE、AURC，按 source、kind、推理/查表子集分组。开发期 temperature 只在 dev-cal 拟合；方案锁定后再用正式 calibration 拟合最终温度。缓存记录模型/adapter、prompt、tokenizer、数据划分的哈希；LoRA 更新后重提特征，不能沿用冻结骨干缓存。

补充开发实验预先固定在 `artifacts/step1/results/protocol.json`：choice 的 train/dev-cal/validation 分别为 4096/512/2048，样本规模曲线使用同一训练序列的 512/2048/4096 前缀；来源轮询且每 group 一条。比较 steps={100,300} × L2={0.001,0.01} 四组，lr=0.01，CE+0.1 Brier。按 group 的确定性 hash 把 validation 分为 select/check；仅 select 的四轮平均 raw NLL 选统一配置，再在 dev-cal 拟合温度并报告 check 与全部 validation。check 仍属开发结果，先前 pilot 已查看部分 validation，不能称为独立测试。

任务分组在观察补充结果前固定：`reasoning-control-v1` 为推理组；`email-selection-control-v1` 和 `ir-control-v1` 为检索/匹配控制组；其余单列 other 并报告来源指标。控制组仍含规则过滤，不是纯查表或配对因果对照；独立 group 很少时明确标记 H3 证据不足。官方 calibration/test/ood 不参与，外部推理集留待后续。配置搜索只保留紧凑指标和最终头；补充实验校验通过后替换旧 pilot 缓存及结果。

### Step 2：E1 final-only

固定四轮 Ouro，只监督第 4 轮；先只训共享 decision head，再以相同数据顺序、步数和种子加入 LoRA r8。评估该头在第 1–3 轮的输出，并训练每轮独立 probe。保留四轮 final-only 和 `total_ut_steps=2` 的 final-only 两个基线。

两轮基线在加载模型前设置 config，并断言 `model.model.total_ut_steps == 2`；只修改加载后的 config 不一定改变实际循环数。E1/E2 从相同预训练权重、head 初始化和训练数据开始，不让 E2 额外接续 E1 训练。记录 optimizer steps、token/候选数、GPU 时间；相同步数的两轮与四轮训练并不等算力，需要另报预算或增加算力对照。

### Step 3：E2 多轮监督

比较 normalized uniform `alpha=[.25,.25,.25,.25]` 与 normalized late-weighted `alpha=[.1,.2,.3,.4]`，固定 E1 数据顺序、步数和种子。核心比较为：E2 第 2 轮 vs 四轮 final-only（浅轮效率）；E2 第 2 轮 vs 两轮 final-only（公平预算）；E2 第 4 轮 vs E1 第 4 轮（深轮是否受损）。

首轮只跑 uniform；有信号后再加 late-weighted 和头形式消融。初期训练矩阵为 E0、四轮 E1、四轮 E2-uniform、两轮 E1，先验证研究主线。

### Step 4：外部与稠密对照

受控稠密基线使用相同数据和训练协议，并记录各 tokenizer 的模板与长度差异；参数和 FLOPs 分别控制，不能把“性能类似 4B”当作“FLOPs 对齐 4B”。Open-Jev checkpoint 遵循其官方 renderer，在同一评测集上作外部参照，并记录它可能见过该训练集的限制；JevK5 只作非受控参考。

### Step 5：决策级退出（可选）

先用全轮缓存离线比较 max-prob、Ouro 最后 token 的门和学习型风险头。当前 Ouro forward 会先计算完全部循环再选择某轮，因此离线退出只能报告质量—理论轮数，不能声称降低实际延迟/FLOPs。若结果支持继续，再改成逐轮 forward，达到退出条件后停止，并测量整道 decision row 的实际 wall-clock/FLOPs。

风险头在 train 中另留 group 子集或用 out-of-fold 预测拟合，不能用 test/ood 拟合。退出阈值在 validation 内再分组划分的 policy-fit 上选取，并在 policy-check 上验证；正式 calibration 只拟合最终温度。Ouro 的 lambda 是条件退出概率，累计退出概率为 `1-product(1-lambda_i)`；直接 lambda 阈值仅作启发式。choice 必须聚合全部候选的信息后作题级退出，不能分别退出后把不同深度且未对齐的分数直接 softmax。区分自适应深度和拒答 coverage，分别报告质量—成本与选择性风险。

## 评测细节

- 候选顺序固定随机种子；记录 prompt 长度、截断率、候选数和实际循环数。
- ECE 同时报告 15 等宽 bin 和自适应 bin；NLL/Brier 用 soft target，Accuracy 只对可定义硬标签的行报告。
- Temperature scaling 的正温度不会改变单样本 argmax；二分类统一温度不改变最大概率排序，多分类可能改变。因此分别解释准确率、校准与风险排序；跨 source/kind/候选数报告总体和分组结果。
- Accuracy/ECE/0-1 AURC 主结果只用 one-hot target；软标签行用 CE/Brier 单列，并可报告 expected risk，不能将 argmax(target) 默认为观测真值。
- 置信区间按 `group_id` bootstrap；等效性需预设允许退化幅度，不能只看 CI 是否重叠。
- 按 in-distribution、ood、source、kind、推理/查表子集报告 Accuracy、NLL、Brier、ECE、AURC、Risk–Coverage、实际延迟和循环数。
