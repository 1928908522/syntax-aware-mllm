# 实验问题与解决记录

本文记录实现过程中影响实验有效性的关键问题。它们不是最终方法的一部分，但解释了当前代码为何采用现有的数据格式、监控指标和默认参数。

## Windows命令续行失败

### 现象

在CMD中使用PowerShell或类Unix续行符，导致首行后的`--adapter`、`--output_jsonl`等参数被当作独立命令。

### 处理

发布版`scripts/*.cmd`中的每条Python命令均保持完整一行，不依赖续行符。所有入口同时提供`--help`，便于先核对参数名。

## DPO reference数据不一致

### 现象

训练引用了不存在的`*_ref_mean.jsonl`，旧数据中部分reference gap与当前chosen/rejected不匹配，导致训练速度和损失解释受到影响。

### 处理

增加`precompute_dpo_reference.py`，从当前E1 checkpoint和当前偏好数据重新计算reference log-probability。DPO监控同时报告policy gap、reference gap、跳过的异常reference样本和坏损失计数。

## 评测输入schema混用

### 现象

评测脚本最初只识别特定参数或旧CoT测试文件；切换到syntax-RL测试集时可能出现“没有可用测试样本”，或者关键关系命中率因目标字段不同而不可比较。

### 处理

统一评测入口读取`input_jsonl`，并从`target_triples / selected_triple`解析目标。训练集与测试集分别构造，held-out评测不回读训练样本。

## RL策略信号率为零

### 现象

早期RL采样中，同一prompt的多个响应完全一致，group reward标准差为0，policy loss为0，`policy_signal_rate=0%`。这种情况下训练虽然运行，但没有有效策略更新。

### 处理

增加多温度探索采样、`top_p / top_k`控制和response/reward去重监控。对同奖励组显式跳过策略更新，并报告`unique_response`、`unique_reward`、`group_std`和`skipped_same_reward`。

## Image Route监督权重过高

### 现象

Image Route是自研模块，尚未证明其排序始终可靠。将其作为主要奖励或目标来源时，模型可能学习路由偏差，而不是文本中可验证的句法差异。

### 处理

RL数据由正负caption重新解析并计算方向敏感差异三元组。Image Route只保留为有界样本权重消融，默认系数为0；数据统计明确报告`text_contrast / route_fallback / answer_only`来源。

## 弱字符串匹配高估Key Relation Hit

### 现象

只检查目标词是否出现在输出中，会把方向错误、关系标签错误或模板复述误判为关键关系命中。

### 处理

评测器先解析`<src, rel, dst>`，再分别计算字段匹配和严格完整匹配。最终同时报告`Strict KeyHit`与`Triple Score`，避免单一宽松指标掩盖错误。

## RL关系增强伴随答案下降

### 现象

5% RL将严格关系命中率从16%提升到50%，但答案准确率从88%下降到86%。逐样本分析显示RL修复4个答案错误，同时新增6个错误。

### 处理

保留chosen SFT约束和独立answer reward，并将答案、三元组、joint、format分别监控。后续平衡实验提高answer reward权重，而不是只根据总reward或RL loss选择模型。

## 训练监控原则

DPO至少监控loss、preference loss、SFT loss、policy/reference gap、生成格式和异常跳过数。RL至少监控reward、answer、triple、joint、format、group std、生成多样性、policy signal rate和gradient norm。任何总损失下降都不能替代held-out任务指标。
