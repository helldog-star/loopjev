# Step5C：未来NLL改善可预测性与轻量退出风险头

本协议在新增group的E2预测生成、风险头拟合和退出阈值选择之前固定。
这是单种子pilot；不增加MLP、不修改Ouro/LoRA/Decision Head、不实现实际动态停止。

## 研究问题与划分

当前及之前轮次的六个标量，能否预测第三轮退出相对第四轮的Accuracy和raw/scaled NLL损失？
取原始train中的新choice group，按来源轮询、固定seed17，每group一题，随机分配为：
Risk-train2048、Risk-val1024、Policy-fit1024、Final-check2048。
排除此前E2训练4096题、dev-cal512题、已查看validation2048题所在group；
额外排除全部dev-cal哈希预留group，避免复用温度预留数据。
冻结E2 Uniform-4和旧dev-cal温度，不重新校准。正式calibration/test/ood不访问。

规范化NFKC/大小写/空白后，对context+question精确排除重复，不依赖候选顺序；
context+question+排序候选文本做5-token shingle、32-signature/8-band LSH近重复候选检索，
Jaccard≥0.9则拒绝后选题。旧数据优先，新选题全局去重，跨四份划分相同规则。
近重复检索有漏检可能；共享问句模板另作数量审计，不声明template-disjoint或跨来源泛化。
只保留K≥2、目标分布合法、所有候选长度≤4096的题，不截断。候选置换沿用seed0。

每份输入manifest在推理前保存哈希。只有Risk-train用于拟合标准化/参数；
Risk-val选正则，不重新拟合train+val；Policy-fit选退出阈值；Final-check等模型及策略锁定后才生成预测。
旧fit/check只作研究背景，不进入四份新数据。所有统计仍限于这个冻结E2单种子和新train留出分布。

## 特征与目标

六项固定特征：第三轮raw confidence、raw margin、raw entropy/log(K)、第二到第三轮raw JS、
argmax是否一致、候选平均的第三轮累计原生Gate。Gate先sigmoid，再逐候选累计`1-prod(1-lambda)`。
全部特征只依赖前三轮；标签、第四轮输出、来源/长度/候选数不作为额外输入。
K=1拒绝入池，避免归一熵除零。

目标：NLL Regret raw/scaled分别为loss3-loss4，有符号且不裁剪。
硬标签另定义三类事件：无Accuracy变化、第四轮纠错、第四轮改错；
soft target不产生硬事件监督，NLL仍覆盖全部题。预测Accuracy Regret=P(纠错)-P(改错)。
不把三个目标经验加权为单一Risk Score。
长尾事件固定为每题raw/scaled Regret>0.5 nats，主报raw，并同时报告scaled。

## C0/C1/C2与小网格

- C0：Risk-train的常数NLL均值、硬标签事件频率。
- C1：raw confidence单特征；两项Ridge回归和三分类Logistic。
- C2：全部六特征；两项Ridge回归和三分类Logistic。

StandardScaler只在Risk-train拟合。每个NLL目标分别用Risk-val MSE选择Ridge alpha∈{0.1,1,10,100}，
并列取较强正则；事件Logistic不加类别权重，C∈{0.1,1,10}，按Risk-val log loss选择，并列取较强正则。
最大迭代2000，未收敛报错；训练只观察一种事件类时用频率兜底。MLP不属于本轮。

## Policy-fit锁定与Final-check封存

仅允许第三轮整题退出，第四轮兜底。分别对C0/C1/C2搜索三个输出阈值的AND规则：
raw和scaled NLL阈值各∈{-0.05,0,0.02,0.05,0.1,0.2,+inf}；Accuracy Regret阈值∈{0,0.01,0.03,0.05,+inf}。
每个模型245个组合，另有always4；+inf在JSON中记为null，整个阈值为null表示always4。
Policy-fit配对group bootstrap1000次、seed17；Accuracy下界>-0.01、两套NLL上界<0.02均满足才可选，
最大化退出题数，并列选scaled NLL增加更少者，仍并列按固定网格顺序；无候选则always4。
额外锁定raw confidence直接阈值基线{0.8,0.85,0.9,0.92,0.94,0.96,0.98,0.99}和always4。
它与C1不同：C1先学习confidence到Regret的映射。

models.joblib、protocol.json、开发指标和三家策略写入locked.json哈希；之后才提取Final-check。
Final-check只复核一次，不在其上选择模型/阈值或返工调整；运行前记录访问标记。

## 指标、对照与Go/No-go

Final-check按group配对bootstrap2000次、seed17。

1. NLL预测：两口径MAE/RMSE；C2相对C0、C1逐题平方误差的配对区间。
   预设有用信号：两口径RMSE均改善至少1%，且平方误差差值的单侧95%上界<0。
   没有通过不能自动视为不存在预测信息，需区分区间宽与点估计没有改善。
2. 长尾：Regret>0.5的数量、按预测Regret最高20%覆盖的Recall、预测值>0.5时的Recall，
   预测十分位对应的真实均值、实际退出集的长尾题数和最大损失。
3. Accuracy事件：纠错/改错各自prevalence、Brier、AUROC、AP、top20% Recall和五等宽校准bin，
   无事件时相关指标记null，不用分类Accuracy作主指标。
4. 退出：最终全部题组成的策略对本模型第四轮的Accuracy/NLL差值及单侧界，
   Accuracy仅one-hot、NLL全部题；同时报告退出数量、平均轮数、理论saving=r/4。
5. 同预算：匹配C2退出数量的最高confidence排序，以及21个预设seed随机选题；不按标签挑随机种子。
   与随机的配对统计采用均匀随机退出的精确期望r*d，不将21个seed当作独立模型种子。
   confidence的top-count是离线预算诊断，非已锁定的在线阈值。另报raw Ridge排序退出10/20/40%的描述性曲线。
   不将这些覆盖比例当成新策略选择，也不声称候选数/token/FLOP或延迟已匹配。

投入真实动态停止的条件：预测信号达到第1项；C2锁定策略在Final-check满足三项非劣且理论saving≥10%；
同预算相对confidence和随机期望的两套NLL点估计均改善至少0.005 nats/row、差值单侧95%上界<0，
且Accuracy差值下界>-0.01。任一不满足则暂不进入动态停止，也不自动扩展MLP。
逐项区间为单种子探索分析，未做多重比较校正，不声称总体家族错误率保证。

## 产物纪律

脚本和中文文档在scripts/step5；manifest和紧凑六维特征/目标在artifacts/step5/risk_uniform4_v1/cache。
不保存新hidden states、完整候选分数或模型复制；risk模型、协议、审计、指标在results。
logs只保留一份简短进度日志；不按批次堆积文件，不覆盖历史Step5A/B结果。
