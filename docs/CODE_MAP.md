# 代码地图

## 贡献一：负文本与测试集

| 步骤 | 模块 | 输入 | 输出 |
|---|---|---|---|
| 依存解析 | `syntax/dependency_parser.py` | caption | nodes与dependency edges |
| 纯句法排序 | `data/extract_triples.py` | captions JSONL | ranked triples JSONL |
| 扰动可行性 | `data/build_candidates.py` | ranked triples | typed candidates |
| 负文本生成 | `data/generate_negatives.py` | caption、triples | generated negatives |
| 质量过滤 | `data/filter_negatives.py` | generated negatives | accepted syntax negatives |
| 公共指标 | `evaluation/text_metrics.py` | positive/negative | TED、Jaccard、semantic sim、NLI |
| 可选视觉排序 | `image_route/rank_targets.py` | teacher trajectories | route-ranked triples |

`extract_triples.py`和`image_route/rank_targets.py`输出相同的核心schema，后续步骤可以不变地比较纯句法与Image Route排序。

## 贡献二：微调框架

| 步骤 | 模块 | 作用 |
|---|---|---|
| CoT模板 | `data/cot_templates.py` | 统一chosen/rejected解释格式 |
| SFT数据 | `data/build_cot_data.py` | 划分train/test并构造回答 |
| 偏好数据 | `data/build_preference_data.py` | 构造三类困难rejected |
| RL数据 | `data/build_syntax_rl_data.py` | 从正负文本差异重建目标三元组 |
| CoT SFT | `training/train_cot_sft.py` | LoRA监督微调 |
| Reference预计算 | `training/precompute_dpo_reference.py` | 缓存reference log-probability |
| DPO | `training/train_dpo.py` | syntax-first偏好优化与监控 |
| Reward | `training/rewards.py` | 严格解析、三元组与联合奖励 |
| RL | `training/train_syntax_rl.py` | 组相对策略优化与监控 |
| Held-out评测 | `evaluation/evaluate_reasoning.py` | Answer/Fmt/KeyHit/TripleScore分解 |

外部相关性评测保留`evaluation/evaluate_mmstar.py`和`evaluation/evaluate_winoground.py`。它们作为补充证据，不替代面向句法三元组的自建held-out测试。

结果汇总保存在`results/experiment_results.json`，`scripts/plot_results.py`负责生成README和实验文档中的全部图表。实验故障、诊断信号与修复依据见`docs/ENGINEERING_NOTES.md`。

## 被移出的历史代码

整理版不包含Stage A-C的诊断脚本、重复benchmark入口、临时数据检查脚本、`__pycache__`、本机数据、权重与输出。它们仍保留在原始`vlm`目录，不影响正在运行或未来追溯实验。
