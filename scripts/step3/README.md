# Step 3：让每一轮都接受监督，争取浅轮也能做好决策

使用说明见本页；正式指标、配对结论与限制见 [result.md](result.md)。

## 要回答什么

Step 2 的四轮 LoRA 较好，但读取其第二轮的质量仍落后。
本步对四轮输出同时监督，检验是否能让第二轮接近第四轮基线，同时不损伤末轮。


## 脚本分工

- `step3_train.py`：多轮共享头训练，默认 LoRA + uniform 四轮监督。
- `step3_compare.py`：读取已有预测，与 Step 2 配对比较。
- `run_step3.sh`：先训练 uniform4，完成后依次运行与 lora4、lora2 的比较。
- `PROTOCOL.md`：原技术协议和详细验证记录；主要使用方法见本中文 README。

Step 3 复用 Step 2 的输入核验、加载、特征提取与评估代码。
跨阶段模块由统一入口配置。


## 固定设置

从原始 Ouro 和全新 Yes-minus-No head 开始，**不接续 Step 2 checkpoint**。
默认四轮权重 [.25,.25,.25,.25]，权重之和为 1；每轮计算 CE+0.1 Brier，
再按权重相加，每道题只计一次，L2 每次更新只加一次。
数据、候选顺序、seed=0、300 步×16 题、学习率与 LoRA 配置均与 E1 对齐。
LoRA 更新后重新提取每轮特征，不能使用原冻结表示替代。

可选 `--weighting late` 对应 [.1,.2,.3,.4]，按计划先看 uniform 信号再决定是否开展。
训练成本要单独报告；四轮 E2 与两轮 E1 的步数相同不等于算力相同。


## 运行方式

以下是独立复现实验示例；OUT必须是未使用的新目录。默认正式结果已在
`artifacts/step3/uniform4/`，无需重新训练。训练和比较统一经 `scripts/run.py` 运行。

```bash
cd /root/siton-tmp/home/liuxinyu/loopjev
PY=/root/siton-tmp/home/liuxinyu/miniconda3/envs/loopjev/bin/python
OUT=artifacts/step3/uniform4_reproduce "$PY" scripts/run.py step3 matrix
```

矩阵入口支持环境变量PY、OUT、REF4、REF2。单条件参数见 `step3 train --help`。
只比较已有预测时使用新的输出文件：

```bash
$PY scripts/run.py step3 compare --candidate artifacts/step3/uniform4   --reference artifacts/step2/lora4 --out artifacts/step3/comparisons/vs-e1-four-loop_recheck.json
$PY scripts/run.py step3 compare --candidate artifacts/step3/uniform4   --reference artifacts/step2/lora2 --out artifacts/step3/comparisons/vs-e1-two-loop_recheck.json
```

## 输入、产物与评测约定

输入：原始Ouro、Open-Jev数据、Step1固定缓存及Step2参考结果。
`results/` 保存协议、adapter、head、probe、逐轮预测、指标与比较；`cache/` 保存重提特征。
共享决策头字段为 `multi_loop_head`，独立probe仅作诊断。
训练成功后仅比较失败时，修复比较即可，不覆盖权重重训。

主非劣标准：Accuracy差值单侧95%下界 > −0.01，且NLL上界 < +0.02。
离线读取第2轮不代表真实只执行2轮；详细实验定义见 [PROTOCOL.md](PROTOCOL.md)。
