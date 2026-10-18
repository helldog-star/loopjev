# Step5：决策级退出

研究按题选择计算轮数能否保留质量、减少平均轮数。Step5A/B复用缓存，不训练或重新提取。
Step5C在新group上生成冻结E2预测，拟合轻量风险头；各阶段均未测量实际提速。
结果和局限见 [result.md](result.md)。

## 脚本与范围

- `offline_exit.py`：固定阈值网格，在policy-fit选择最大概率/累计门值策略，在policy-check复核。
- `test_offline_exit.py`：验证因果顺序、候选置换不变性、题级退出及门值变换。
- `analyze_exit.py`：第三/四轮纠错、退出信号排序和不可部署的理想选择器参照。
- `test_analyze_exit.py`：风险符号、软标签分母、未来信息隔离和理想参照约束检查。
- `joint_exit.py`：仅允许第三轮退出的少量联合阈值和同预算随机对照。
- `test_joint_exit.py`：联合条件、退出轮次和随机抽样语义检查。
- [PROTOCOL.md](PROTOCOL.md)：固定方法、数据隔离、质量约束与Go/No-go。

Step5C轻量风险头见[RISK_PROTOCOL.md](RISK_PROTOCOL.md)；真正停止计算的推理尚未实现。
阶段目标不能用离线轮数代替真实延迟。

## 运行

使用loopjev主环境；正式输出必须不存在。只读取既有E2、E1预测与E2门值缓存。

```bash
cd /root/siton-tmp/home/liuxinyu/loopjev
PY=/root/siton-tmp/home/liuxinyu/miniconda3/envs/loopjev/bin/python
$PY scripts/run.py step5 test
$PY scripts/run.py step5 offline --out-dir artifacts/step5/offline_uniform4_v1
$PY scripts/run.py step5 test-analysis
$PY scripts/run.py step5 analyze --out-dir artifacts/step5/diagnostics_uniform4_v1
$PY scripts/run.py step5 test-joint
$PY scripts/run.py step5 joint --out-dir artifacts/step5/joint_exit_uniform4_v1
```

默认`--run artifacts/step3/uniform4`，参考`--reference artifacts/step2/lora4`；
默认bootstrap1000次。`results/`保存protocol.json、metrics.json和紧凑exit_decisions.json。
不复制模型或大张量，正式calibration/test/ood不参与。

Step5B同样使用既有E2预测和门缓存，不读取E1；输出protocol.json、metrics.json和diagnostics.png。
只作第三轮相对第四轮的描述性诊断，不选择新退出策略。

Step5B联合实验只允许loop3退出，使用confidence、跨轮argmax稳定性和JS散度的紧凑组合；
在policy-fit锁定一个策略，在policy-check报告策略和同退出数量的固定种子随机策略。

## Step5C：未来NLL收益预测与轻量风险头

- `risk_core.py`：六项因果特征、Ridge/Logistic、配对统计和多约束退出规则。
- `risk_data.py`：新group排除与重复审计、manifest、冻结E2紧凑特征提取。
- `risk_fit.py`：常数/Confidence-only/Full-feature模型选择、策略锁定、一次封存复核。
- `run_risk.sh`：依次提取开发划分、拟合锁定、提取Final-check并评估。
- `test_risk.py`：目标、因果隔离、候选置换、训练标准化与合成端到端检查。

使用loopjev环境；prepare的ROOT必须不存在。固定使用旧E2权重和温度，不新训骨干。
先测试和准备数据；prepare审计成功后才启动GPU阶段，不访问正式calibration/test/ood。

```bash
$PY scripts/run.py step5 test-risk
$PY scripts/run.py step5 risk-data prepare --root artifacts/step5/risk_uniform4_v1
ROOT=artifacts/step5/risk_uniform4_v1 $PY scripts/run.py step5 risk-all
```

四份group用途、样本量、近重复限制、特征和Go/No-go以RISK_PROTOCOL.md为准。
cache保存manifest和六维特征/监督，results保存协议、审计、轻量模型、锁定文件和指标。
可分步运行`risk-data extract`、`risk-fit fit`、`risk-data extract --partitions final_check`、`risk-fit check`；
禁止覆盖已有cache、重复拟合或反复查看Final-check后调参。Final-check提取必须在锁定之后。
