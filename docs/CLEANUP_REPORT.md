# 重构说明

## 原目录的主要问题

- Stage A-E历史脚本、诊断脚本与最终方法入口混放。
- 多个模块通过运行时路径注入绑定原始工程绝对路径。
- 模型、数据、checkpoint和输出目录硬编码为单台机器路径。
- 句法相似度/NLI实现藏在旧多模态对比脚本中，过滤模块反向依赖实验脚本。
- Image Route实验与最终syntax-first训练在命名上混淆。
- 本地API凭据曾有写入源码默认值的风险。

## 整理动作

- 只保留两项贡献的主流程及MMStar、Winoground补充评测。
- 将Stage D/E编号改为职责清晰的`data / training / evaluation`目录。
- 抽出`evaluation/text_metrics.py`作为统一句法相似度与NLI实现。
- 新增`data/extract_triples.py`，建立不依赖Image Route的主数据入口。
- 将Image Route集中到`image_route/`并明确为可选消融。
- 移除全部运行时`sys.path`修改和源码中的本机绝对默认路径。
- 模型使用环境变量或公开模型ID默认值；数据和checkpoint使用项目相对路径。
- API key只允许通过参数或环境变量提供。
- 排除缓存、大型数据、权重、输出和历史诊断文件。

## 未删除的内容

原始研究工作区没有被修改或删除，仍可用于追溯历史实验。所谓“删除冗余”只发生在精简发布版中，即不复制与最终贡献无关的文件。

## 验证

- 全部Python文件通过AST语法解析。
- 核心奖励与三元组逻辑9项单元测试通过。
- 数据构造、SFT、DPO、RL、held-out、MMStar、Winoground和Image Route入口均通过`--help`导入检查。
