# 实验结果

所有图表均由`results/experiment_results.json`和`scripts/plot_results.py`生成，表格与图片共用同一份数据源。

## 构造方式对比

| 方法 | n | Contradiction | TED | Jaccard | Semantic sim | Combined |
|---|---:|---:|---:|---:|---:|---:|
| TopK triple | 10000 | 84.80% | 1.703 | 0.582 | 0.839 | 0.766 |
| Random triple | 9988 | 74.80% | 6.318 | 0.116 | 0.577 | 0.394 |
| All triples | 9987 | 74.86% | 6.273 | 0.109 | 0.585 | 0.398 |
| No triple | 9982 | 72.24% | 6.099 | 0.149 | 0.604 | 0.423 |

![负文本构造方法对比](../assets/figures/negative_construction_comparison.png)

### 指标解释

- `Contradiction`：NLI判断正文本与负文本构成矛盾的比例，越高越符合困难负样本目标。
- `TED`：依存树编辑距离，越低表示句法树变化越小。
- `Jaccard`：依存边集合Jaccard，越高表示共享句法关系越多。
- `Semantic sim`：句向量余弦相似度，用于约束整体表达不要偏离过远。
- `Combined`：归一化TED、Jaccard和语义相似度的加权综合分数。

Top-K相较Random将contradiction提高10.0个百分点、TED降低4.615、Jaccard提高0.466。这支持“选择少量关键三元组进行定向扰动”比随机、全量提示或不提供关系更适合本任务。

## Syntax-target held-out 100

| 模型 | Answer | Format | Strict KeyHit | Triple score |
|---|---:|---:|---:|---:|
| 部分训练DPO | 88% | 100% | 16% | 0.291 |
| 5% syntax-first RL | 86% | 100% | 50% | 0.712 |

![Held-out模型对比](../assets/figures/heldout_model_comparison.png)

逐样本比较中，RL实现37个miss-to-hit和3个hit-to-miss；答案方面修复4个DPO错误，同时新增6个错误，净下降2个百分点。

### 指标解释

- `Answer`：最终A/B答案正确率。
- `Format`：响应能否按约定格式解析。
- `Strict KeyHit`：预测中是否包含至少一个`src / rel / dst`均正确且方向一致的目标三元组。
- `Triple score`：三元组字段部分匹配得分，用于区分完全错误与局部正确。

## 解读与限制

结果显示syntax-first RL显著提高目标三元组表达，但目前不是无条件整体胜出：答案准确率略有下降，样本仅100条，且训练主要来自MSCOCO约1万样本。DPO只完成约1万偏好对训练，RL结果来自5%子集，不能外推为完整训练上限。后续优先扩大held-out评测并调整答案奖励系数，再扩大数据来源、训练样本和epoch。
