# Step 4 实验协议

本文件记录固定方法和比较边界；运行命令见README，结果见result.md。

## 模型、提示词和决策方式

模型：项目下 `models/Qwen3-1.7B-Base`，28 层、隐藏维度 2048。
只更新 q_proj/v_proj，LoRA rank=8、alpha=16、dropout=0。
实际加载后必须断言 adapter 参数数为 1,605,632；标量 head 为 2048 参数、无 bias。
Ouro Q/V r8 的 adapter 为 1,572,864，约差 2.08%，不是完全相同参数量。

使用经过原项目核验的 candidate renderer：只含 state、question 和当前候选。
对于 Base 模型，采用固定纯文本包装，在候选提示末尾加 `\nAnswer:\n`，
**不调用 chat template，不生成思维链或答案 token**。
读取最后一个非 padding token 的最终隐藏态，用共享标量头给各候选打分，
在整道题所有候选之间 softmax，再计算 CE + 0.1 Brier。
候选 microbatch 只控制显存，不把一道题拆成多个独立分类损失。

head 使用本模型输出层的裸词 Yes−No 行初始化；本地 tokenizer 对应 ID 为 9454/2753。
预检逐候选验证真实后缀下追加 Yes/No 的 token 边界，失败即停止；不能复制 Ouro 的 ID。
记录初始化 head 范数，披露不同模型输出行尺度可能不同。


## 固定数据和训练预算

- 沿用 Step 1 train/dev-cal/validation=4096/512/2048 的 group、题目、候选置换与标签。
- 校验原 Ouro 模型/缓存、数据文件哈希和原始 candidate prompt 哈希，防止换题或渲染变化。
- Qwen 重新编码，并记录自身提示词哈希、长度及源模型长度；不能沿用 Ouro 隐藏态。
- 最大长度和单 microbatch token 预算默认均为4096；超长直接报错，禁止静默截断或删题。
- 默认 seed=0、300 optimizer updates、16题/update；head LR=.001、LoRA LR=.0002、
  raw-head L2=.001、AdamW无weight decay、梯度裁剪1，non-reentrant checkpoint。
- 正式 calibration/test/ood 不参与本步训练、温度拟合或开发评估。

模型预训练不同，tokenizer 与提示包装不同；等题数/步数不等于等token、等FLOPs或等时间。
记录实际 token 数、训练时间、显存、可训练参数数和预分词输入下的推理延迟。
只读 CPU 预检会校验大文件哈希和全部候选，可能需要数分钟。


## 外部基线与校准

Open-Jev保留发布adapter、独立标量head、官方renderer和chat template，enable_thinking=False。
固定官方推理源码提交 `bd4118882f733574a3250a4b65fe4d884130c08b`，完整源码位于 `repos/Open-Jev/`；加载时校验Git提交及工作区差异，
每次评测协议记录实际源码文件哈希。迁移核验见
`artifacts/step4/source_migration/results/audit.json`。固定基础模型版本见README与openjev-base-manifest.json。
不重新训练、不生成答案、不静默截断；输入超过4096 tokens时拒绝。
仅评测dev-cal512题和validation2048题，模型输入不含target。

分别报告温度1、发布温度1.518796342858676、本项目dev-cal重新拟合温度。
重新拟合直接作用于raw logits，不能再次叠加发布温度。
比较方向为候选模型减参考；raw_vs_raw、dev_cal_vs_dev_cal、released_vs_raw分别解释。
外部checkpoint历史训练量为80,816行，不能排除训练数据重合，不视作等预算对照。

## 比较边界

Qwen/Ouro对齐样本、候选顺序、训练调度、损失与更新次数，不强称预训练、token或FLOPs相等。
协议核验不能通过修改历史哈希或放宽样本配对来绕过。
Ouro参考模型实际运行4轮时，读取第2轮仍计完整4轮延迟，不是实际提前退出成本。
Open-Jev计时含分词、padding和传输，与预分词Qwen/Ouro计时不可直接求速度比。
指标包括raw/scaled准确率、NLL、Brier、两种ECE、选择性风险及任务/来源分组。
ECE衡量置信度与准确率偏差；AURC为风险—覆盖率曲线下面积。
单种子group bootstrap不包含训练种子波动，validation-check仍是开发划分。

## 基座训练阶段的混杂因素

Open-Jev使用的Qwen3.5-2B是后训练模型（官方说明：https://huggingface.co/Qwen/Qwen3.5-2B），
而本项目Ouro与Qwen3-1.7B使用Base权重。后训练可能影响任务理解、表示与概率校准，
因此Open-Jev只作外部系统参考，不能据其差异归因循环结构或多轮监督收益。
不能把发布adapter换接Base基座来模拟公平对照；若单独检验后训练作用，
应另设同一家族Base/后训练权重、相同下游训练与评测的配对实验。
