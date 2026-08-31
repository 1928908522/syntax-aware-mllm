# 复现指南

## 环境

建议Python 3.10、CUDA 12.x和项目现有`vlm` Conda环境。安装项目后下载spaCy模型：

```powershell
pip install -e .
python -m spacy download en_core_web_lg
```

如模型已在本地，设置`QWEN_VL_MODEL`、`NEGATIVE_GENERATOR_MODEL`、`SYNTAX_EMBEDDING_MODEL`和`SYNTAX_NLI_MODEL`。不要把API key写入源码或提交到版本库。

## 输入格式

`data/captions.jsonl`每行至少包含：

```json
{"image_path":"D:\\dataset\\images\\0001.jpg","caption":"A person is holding a red umbrella."}
```

## 完整顺序

1. 运行`scripts/01_build_dataset.cmd`构造负文本、train/test、DPO与RL数据。
2. 运行`scripts/02_train.cmd`依次训练SFT、预计算reference、DPO和RL。
3. 运行`scripts/03_evaluate.cmd`在同一held-out集合比较三个checkpoint。

这些脚本没有使用CMD续行符，每条命令均为完整一行，便于逐条粘贴或中断。

## 5%冒烟实验

先在各训练命令追加`--sample_fraction 0.05`。RL建议保留`--monitor_every 25 --monitor_generate_every 100`，DPO建议保留`--monitor_every 100 --monitor_generate_every 500`。确认格式、policy signal、reward差异和梯度均正常后再扩大样本。

## 评价指标

- `Answer Acc`：最终A/B答案正确比例。
- `Format OK`：响应满足可解析格式的比例。
- `Key relation hit`：预测中至少出现一个完整且方向一致的目标三元组。
- `Key relation score`：`src / rel / dst`部分匹配的平均分。
- `Answer+Triple`：答案正确且严格三元组命中，用于衡量联合能力。
