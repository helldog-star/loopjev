# Step 4：稠密模型与外部模型对照

比较Qwen稠密模型与Ouro E1/E2，并加入Open-Jev发布模型作为外部参考。
方法定义见 [PROTOCOL.md](PROTOCOL.md)，实验状态、证据和结论见 [result.md](result.md)。

## 两类模型

- 受控下游基线：`models/Qwen3-1.7B-Base`，Q/V LoRA r8、alpha16，300步×16题。
  沿用Step1固定样本与Step2优化设置；“受控”不意味着预训练或算力完全相同。
- 外部基线：`models/Open-Jev-2B/package`，只有adapter和决策头，必须配套
  `Qwen/Qwen3.5-2B`，revision `15852e8c16360a2fea060d615a32b45270f8a8fc`。
  不可用Qwen3-1.7B替代；只推理，不训练。

## 脚本分工

- `dense_train.py`、`dense_compare.py`、`run_dense.sh`：Qwen预检、训练、评测与比较。
- `openjev_eval.py`、`openjev_compare.py`：发布包审计、官方模型推理与外部比较。
- `setup_openjev_env.sh`：创建独立Open-Jev环境；不升级原loopjev环境。
- `test_dense.py`、`test_openjev.py`：CPU回归测试。
- `openjev-base-manifest.json`：固定基座文件校验元数据。

## Qwen运行顺序

命令从项目根目录执行，所有输出目录必须未被使用。先预检、再GPU空闲时冒烟、最后正式训练。
预检目录不能直接用作训练目录；微型CPU测试不能代替真实权重冒烟。

```bash
cd /root/siton-tmp/home/liuxinyu/loopjev
PY=/root/siton-tmp/home/liuxinyu/miniconda3/envs/loopjev/bin/python
$PY scripts/run.py step4 test-dense
$PY scripts/run.py step4 train --preflight-only --out-dir work/step4_input_audit
$PY scripts/run.py step4 train --smoke --steps 1 --batch-rows 1 --bootstrap 20 \
  --benchmark-rows 1 --benchmark-repeats 1 --out-dir work/step4_gpu_smoke
# 冒烟通过后，确认默认正式输出尚不存在再运行：
nohup "$PY" scripts/run.py step4 matrix > logs/step4_qwen.log 2>&1 &
```

默认正式目录 `artifacts/step4/qwen3_1p7b_qv_r8`；矩阵入口支持环境变量PY、OUT。
矩阵会比较E1两轮/四轮和已就绪的E2第二/四轮。比较失败时只补跑比较，不覆盖训练结果。

```bash
$PY scripts/run.py step4 compare --candidate artifacts/step4/qwen3_1p7b_qv_r8 \
  --reference artifacts/step2/lora4 --out artifacts/step4/comparisons/qwen-vs-e1-four-loop.json
```

## Open-Jev运行顺序

下载完整指定版本基座、创建独立环境后再审计及推理。安装脚本拒绝覆盖已有环境。

```bash
hf download Qwen/Qwen3.5-2B \
  --revision 15852e8c16360a2fea060d615a32b45270f8a8fc --local-dir models/Qwen3.5-2B
$PY scripts/run.py step4 external-env
JEVPY=/root/siton-tmp/home/liuxinyu/loopjev/.envs/openjev/bin/python
$PY scripts/run.py step4 test-external
$JEVPY scripts/run.py step4 external-eval --audit-only --out-dir work/openjev_audit_recheck
# 审计就绪且GPU空闲时：
$JEVPY scripts/run.py step4 external-eval --smoke --benchmark-rows 1 \
  --benchmark-repeats 1 --out-dir work/openjev_gpu_smoke
# 冒烟通过后：
$JEVPY scripts/run.py step4 external-eval --out-dir artifacts/step4/openjev2b
$PY scripts/run.py step4 external-compare --candidate artifacts/step4/openjev2b \
  --reference artifacts/step2/lora4 --out artifacts/step4/openjev2b/results/vs-e1-four-loop.json
```

官方要求torch>=2.8；当前独立环境为torch2.14.1+cu130，
Transformers5.10.2、PEFT0.19.1、Accelerate1.13.0；评测协议记录实际版本。
正式推理离线加载，脚本验证文件哈希；`--base-model`可指定另一个已下载位置。
比较E2时使用 `--reference artifacts/step3/uniform4 --reference-loop 2`（或4），
比较Qwen时指定其正式目录，并使用新的 `--out` 文件。

## 输入与产物

公共输入是Step1固定缓存和原始数据；Qwen重新分词并在更新LoRA后重新提取特征。
`results/`保存协议、指标、验证预测；Qwen另存adapter、head、probe与`cache/`特征。
Open-Jev不复制权重或保存隐藏态缓存。审计和冒烟只作验证，不能作为正式性能结论。
质量报告保留原始与校准后两套指标；外部基线还单列发布温度条件。
完整比较限制和计时口径见协议，数值分析集中在result.md。

## 串行执行完整对比

真实Qwen冒烟通过、Open-Jev独立环境已准备后，可用 `step4 all` 串行运行Qwen矩阵、
Open-Jev真实权重冒烟、外部完整评测以及五项配对比较。拒绝覆盖已有输出或日志；
任一步失败即停止。已有部分结果时改用上面的单项命令继续，不重新训练。

```bash
nohup "$PY" scripts/run.py step4 all > logs/step4_pipeline.log 2>&1 &
```

主日志为 `logs/step4_pipeline.log`，子任务日志为 `step4_qwen.log`、
`step4_openjev_smoke.log`、`step4_openjev_eval.log`（均在logs目录）。

## 官方源码位置

统一使用完整仓库 `repos/Open-Jev/`，固定提交
`bd4118882f733574a3250a4b65fe4d884130c08b`，工作区保持干净。
`external-eval` 默认从该目录加载；无需另外保留参考源码或推理模块副本。
更新上游版本须作为明确的新实验变更，不直接拉取最新版覆盖现有协议。
