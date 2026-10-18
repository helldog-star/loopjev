# Step 4 结果分析：稠密与外部模型对照

核对日期：2026-10-18。**尚无正式性能结论**。
已存档的证据只有输入预检与发布包审计；未发现本阶段正式训练/推理指标。
预检通过不等于模型性能达标，也不等于真实权重GPU冒烟通过。

## Qwen3-1.7B-Base：输入预检

依据 [preflight.json](../../artifacts/step4/input_audit/results/preflight.json)，状态为 `CPU PREFLIGHT ONLY`：

- train/dev-cal/validation = 4096/512/2048题，候选数26414/3324/9216。
- 三个划分均未截断；最长输入分别1265/1265/1224 tokens。
- 未padding的token总数为15,980,872 / 1,979,445 / 3,938,998，
  这是完整数据遍历的计数，不是300步训练消耗。

结论仅限输入可按既定划分与候选顺序处理。随机微型模型CPU回归通过，
真实1.7B权重单步GPU冒烟已通过（每split8题，SMOKE ONLY）；
仍需正式训练和配对比较，才能分析质量与成本。

## Open-Jev-2B：包审计与样本对齐

依据 [audit.json](../../artifacts/step4/openjev_audit/results/audit.json)，状态为 `AUDIT ONLY`：

- 发布包文件大小与SHA256检查通过。
- dev-cal 512题/3324候选、validation 2048题/9216候选与原缓存语义及顺序对齐。
- 该次审计记录 `ready_for_inference=false`：当时固定Qwen3.5-2B基座及独立环境尚未就绪。
  这是一份历史审计，不代表下载完成后的实时状态；准备完毕后应重新审计。

尚无真实checkpoint推理指标、温度拟合结果或配对性能结论。
官方发布包的历史温度1.518796342858676不是本项目新测出的最优温度。

## 正式结果应回答什么

1. 同样下游样本、更新步数和Q/V LoRA设置下，Qwen与Ouro E1/E2质量、成本如何？
2. Open-Jev发布checkpoint在相同评测样本上的表现如何？它只作为外部参考。
3. 分别报告raw、dev-cal校准和（外部模型）发布温度条件，不选择性报告有利口径。

训练数据、预训练、模型结构和tokenizer并未完全控制，不能将跨模型差异全归因于循环结构。
Open-Jev训练量和潜在数据重合不同，不能当作等预算受控消融。
外部模型计时包含分词和传输，现有Ouro/Qwen计时基于预分词输入，不直接计算速度比。

## 下一步

核验模型与环境 → GPU空闲时真实权重冒烟 → 正式运行 → 按现有配对规则生成比较。
正式结果到齐后，在本文件追加对应实验条件、样本数、种子、指标与区间；运行命令见README。

## 本轮推进状态

Qwen3.5-2B下载文件已通过指定revision的文件哈希核验；独立环境已就绪。
Qwen真实权重单步冒烟已完成，不能将其指标当作正式结果。
Open-Jev使用后训练基座，与Base训练条件不同，结果只解释为外部系统对照。

Qwen正式串行任务已启动，输出目录为 `artifacts/step4/qwen3_1p7b_qv_r8`；
当前运行状态查看 `logs/step4_pipeline.log` 与 `logs/step4_qwen.log`，尚无正式结果。
Open-Jev在Qwen完成后才执行；依赖检查或冒烟失败会停止队列，不代表外部评测已完成。

## Open-Jev运行前审计

[新审计](../../artifacts/step4/openjev_ready_audit/results/audit.json)的
`ready_for_inference=true`，没有阻塞项；这仅表示文件和依赖检查通过，不是GPU性能结果。
使用完整官方仓库 `repos/Open-Jev/`；[源码核验](../../artifacts/step4/source_migration/results/audit.json)
确认与此前固定推理模块逐文件一致。独立环境实际torch版本为2.14.1+cu130，后续协议按实记录。
