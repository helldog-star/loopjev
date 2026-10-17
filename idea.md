# Adaptive Latent Computation for Accurate and Calibrated Decisions

## Research Question

**Can additional recurrent computation improve decision quality and probability calibration without explicit chain-of-thought generation?**

研究在不生成显式 CoT 的情况下，是否能够通过增加模型内部的循环计算（latent computation）提高决策准确性和概率校准质量，并按需分配计算预算。

## Motivation

- Jev-style 模型直接对给定候选输出概率分布，不必自回归生成答案；Open-Jev 提供了一个可训练的候选打分实现。
- Ouro 通过参数共享的循环计算增加有效计算深度，为“多算几轮再决策”提供了合适的预训练骨干。
- **更多 Loop 不必然意味着更好的 Calibration**：Accuracy、NLL、Brier 和 ECE 可能随深度呈现不同趋势。这正是需要验证的假设，而非预设结论。

## Proposed Method

以 **Ouro-1.4B + Multi-loop Decision Head** 为主线，借鉴 Open-Jev 的 *candidate-wise scoring* 训练范式。

1. 对每个问题及其候选答案构造输入，经 Ouro 进行多次共享参数循环。
2. 在每次 Loop 后提取隐藏表示，使用**共享的 Scalar Decision Head** 给每个候选打分，再对候选维度做 Softmax：

   \[
   s_k^{(t)}=w^\top h_k^{(t)}+b,\qquad
   p_k^{(t)}=\operatorname{softmax}_k(s^{(t)}).
   \]

3. 比较两种训练目标：
   - **Final-loop supervision**：只监督最后一轮的决策分布。
   - **Multi-loop supervision**：监督各轮输出，使模型具备 *anytime decision* 能力。

   \[
   \mathcal L=\sum_{t=1}^{T}\alpha_t\left[\mathrm{CE}(p^{(t)},y)+\lambda\,\mathrm{Brier}(p^{(t)},y)\right].
   \]

4. 若前述实验有效，再探索 **calibration-/risk-aware adaptive exit**，根据预测错误风险决定是否继续循环；避免仅根据最大概率提前退出。

> 本方案使用监督式概率训练，**不宣称复现官方 Jev 未公开的 RLCD**；也不生成自回归 reasoning trajectory。

## Minimal Experiments (2–3 Weeks)

| 实验 | 配置 | 目标 |
|---|---|---|
| E0 | Frozen Ouro；各 Loop 的 Linear Probe | 检查循环表示是否包含可读出的决策信息 |
| E1 | Ouro + Final-loop Decision Head（LoRA） | 建立可训练基线 |
| E2 | Ouro + Multi-loop Supervision（Uniform / Late-weighted） | 验证中间轮监督的作用 |
| E3 | Open-Jev 或 JevK5 开源 Checkpoint | 外部决策模型基线 |
| E4（可选） | Risk-aware Early Exit | 探索校准—准确率—计算成本权衡 |

**评测：** Accuracy、NLL、Brier Score、ECE、Risk–Coverage，以及每次决策的延迟 / FLOPs。报告各 Loop 的 **Raw** 和 **Temperature-scaled** 指标；温度只在独立 Calibration Split 上拟合。

**关键控制：** 固定数据划分、候选集和训练预算；比较 Final-only 与 Multi-loop 在**同一 Loop Depth** 的表现，避免将额外训练步数或计算量误判为方法收益。对于 Ouro 的 1–4 次循环，注意其原始预训练对最终 Loop 的偏好。

## Fast-track Plan

- **Week 1：** 阅读 Ouro / Open-Jev 核心实现；提取每轮 Hidden States；整理 Decision Row；完成 Linear Probe 和 Final-loop Baseline。
- **Week 2：** 完成 Multi-loop 训练、逐轮评测、Temperature Scaling、主要消融和 OOD 测试；第 14 天作 Go/No-go 决策。
- **Week 3（可选）：** Adaptive Exit、计算预算对照、重复种子、论文写作与代码整理。

## Expected Contribution

揭示循环计算深度对非自回归决策模型的 **accuracy–calibration–compute trade-off**；如果 Multi-loop Supervision 能让更浅的循环达到同等质量，则进一步证明其对高效、自适应决策的价值。

## References / Code

- [Ouro-1.4B (model weights)](https://huggingface.co/ByteDance/Ouro-1.4B)
- [Open-Jev (training and decision-scoring reference)](https://github.com/Zefan-Cai/Open-Jev)
- [Open-Jev training documentation](https://github.com/Zefan-Cai/Open-Jev/blob/main/docs/multidomain-training.md)
- [JevK5 (alternative baseline)](https://github.com/allebee/jevk5)
