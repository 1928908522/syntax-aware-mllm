# 方法设计

## 数据构造

给定图像 `I` 与正确描述 `c+`，解析依存关系集合 `T(c+)`。目标是生成 `c-`，满足：

- `c-`与图像事实冲突；
- `c-`与`c+`在词汇和句法上尽量接近；
- 差异可定位到一个或少量方向敏感三元组。

默认纯句法排序按主客体、谓词、属性和空间关系优先。可选Image Route排序会比较teacher route与image route，但只改变候选优先级，不改变后续过滤标准。

综合相似度定义为：

```text
0.25 * (1 - normalized_TED)
+ 0.25 * dependency_Jaccard
+ 0.50 * embedding_similarity
```

NLI用于描述语义关系分布，文本合法性作为硬过滤，句法与NLI指标作为质量统计和样本权重依据。

## CoT SFT

模型输入图像及两个高度相似caption，输出关键关系、视觉核验、差异说明和最终答案。SFT先稳定格式与解释模式，避免后续偏好/RL从近乎随机的输出空间开始。

## Syntax-first DPO

每个正确响应对应三类rejected：错误答案、错误关键关系、错误视觉支持。DPO目标比较policy相对reference对chosen/rejected的偏好差，同时保留少量chosen SFT损失。训练监控包括偏好损失、SFT损失、policy gap、reference gap、生成格式和跳过计数。

## Syntax-first RL

对每个prompt采样多个响应，在组内标准化奖励形成优势。奖励由以下部分组成：

```text
R = a * answer_correct
  + b * direction_sensitive_triple_score
  + c * answer_and_exact_triple
  + d * format_score
  - missing/non_ascii penalties
```

三元组部分分别匹配`src / rel / dst`，因此不会把方向相反的关系当作命中。句法质量可作为有界样本权重，Image Route权重默认0。监控额外覆盖reward、answer、triple、joint、group std、unique responses、policy signal rate和gradient norm，用于提前识别采样无差异、奖励坍缩或梯度异常。
