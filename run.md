# MichiganCast 运行手册（run.md）

这份文档用来解决 3 件事：

1. 你第一次接手项目时，能按顺序跑通。
2. 你知道每个模块代码在做什么，为什么要做。
3. 你能把每一步的输入、输出和结果讲清楚（用于面试/portfolio）。

---

## 0. 先准备环境

项目绑定 conda 环境 `pytorch_env`。

```bash
source scripts/activate_pytorch_env.sh
```

或不激活 shell，直接包一层运行：

```bash
scripts/run_in_pytorch_env.sh python -V
```

建议后续命令统一用：

```bash
scripts/run_in_pytorch_env.sh <你的命令>
```

---

## 1. 项目目标定义（先统一口径）

代码位置：

- `configs/task.yaml`

主要目的：

- 统一预测任务定义：不是“同一时刻分类”，而是“未来 `t+6h / t+24h` 预测”。
- 明确防泄漏规则：输入窗口结束时刻必须早于标签时刻（`X_time < y_time`）。
- 明确时间切分规则：按年份切 train/val/test。

为什么这一步重要：

- 后续所有清洗、建模、评估都依赖这个定义；定义错了，后面结果都不可信。

---

## 2. 模块 A：数据契约检查（Data Contract）

代码位置：

- `src/data/contracts.py`

这个模块做什么：

- 检查字段是否齐全（列名是否缺失）。
- 检查数值是否在合理物理范围内（比如压强、湿度）。
- 检查缺失比例是否超阈值。
- 输出可定位行号的 JSON 报告。

为什么做：

- 在进入清洗和训练前，先知道原始数据是否“合格”。

如何运行：

```bash
scripts/run_in_pytorch_env.sh python -m src.data.contracts \
  --input data/processed/tabular/traverse_city_daytime_meteo_preprocessed.csv \
  --output artifacts/reports/data_contract_report.json
```

输入：

- `data/processed/tabular/traverse_city_daytime_meteo_preprocessed.csv`

输出：

- `artifacts/reports/data_contract_report.json`

成功判据：

- 看 `status` 字段。`pass` 为通过，`fail` 为不通过。
- 注意：当前实现即使 `status=fail`，脚本也可能 `exit code 0`，这是“校验失败”不是“程序崩溃”。

---

## 3. 模块 B：数据质量检查（Data Validation）

代码位置：

- `src/data/validate.py`

这个模块做什么：

- 检查缺失码（`m/M/NC` 等）。
- 检查时间序列连续性（是否有异常时间间隔）。
- 检查图像目录库存（空文件、缺号、非数字命名）。
- 检查图像和表格是否对齐（行索引 vs 图像 ID）。

为什么做：

- 这一步专门发现“契约之外”的工程问题，尤其是图像-表格错位。

如何运行：

```bash
scripts/run_in_pytorch_env.sh python -m src.data.validate \
  --input data/processed/tabular/traverse_city_daytime_meteo_preprocessed.csv \
  --image-dir data/processed/images/lake_michigan_64_png \
  --output artifacts/data_quality_report.json
```

输出：

- `artifacts/data_quality_report.json`

---

## 4. 模块 C：可复现清洗流水线（Cleaning Pipeline）

代码位置：

- `src/data/clean.py`

这个模块做什么：

- 统一缺失码并转为标准空值。
- 将气象字段转数值类型。
- 把超物理范围值替换为空值。
- 添加统一时间戳并排序。
- 去重、删关键列空值行。
- 输出清洗后数据和清洗摘要。

为什么做：

- 把 notebook 手工清洗变成可复现脚本，确保每次训练输入一致。

如何运行：

```bash
scripts/run_in_pytorch_env.sh python -m src.data.clean \
  --input data/processed/tabular/traverse_city_daytime_meteo_preprocessed.csv \
  --output data/interim/traverse_city_daytime_clean_v1.csv \
  --summary artifacts/reports/data_cleaning_summary.json
```

输出：

- 清洗后数据：`data/interim/traverse_city_daytime_clean_v1.csv`
- 清洗摘要：`artifacts/reports/data_cleaning_summary.json`

---

## 5. 模块 D：EDA 自动报告

代码位置：

- `src/analysis/eda_report.py`

这个模块做什么：

- 统计类别分布（是否严重不平衡）。
- 按月分析降水正类比例（季节性）。
- 计算年度漂移（跨年变化）。
- 输出相关性热图和分布图。

为什么做：

- 给建模前的特征理解、类别不平衡策略、阈值策略提供依据。

如何运行：

```bash
scripts/run_in_pytorch_env.sh python -m src.analysis.eda_report \
  --input data/interim/traverse_city_daytime_clean_v1.csv \
  --summary artifacts/reports/eda_summary.json \
  --fig-dir artifacts/figures
```

输出：

- `artifacts/reports/eda_summary.json`
- `artifacts/figures/eda_*.png`

---

## 6. 模块 E：标签构造与时间切分（防泄漏核心）

代码位置：

- `src/features/labeling.py`
- `src/data/split.py`

这个模块做什么：

- 生成未来预测样本索引（`t -> t+h`）。
- 硬性校验 `X_time < y_time`，阻止同刻信息泄漏。
- 按年份做时间切分（train/val/test），而不是随机切分。

为什么做：

- 这是“课设风格”到“生产风格”的关键：时间任务必须用时间切分，避免未来信息泄漏。

如何运行（切分 CLI）：

```bash
scripts/run_in_pytorch_env.sh python -m src.data.split \
  --input data/interim/traverse_city_daytime_clean_v1.csv \
  --output-dir data/interim/splits \
  --train-years 2006:2012 \
  --val-years 2013:2014 \
  --test-years 2015:2015
```

说明：

- `labeling.py` 目前主要被基线模型和多模态训练入口自动调用（不是独立 CLI 流程）。

---

## 7. 模块 F：传统机器学习基线（Baseline）

代码位置：

- `src/models/baselines.py`
- `src/eval/metrics.py`

这个模块做什么：

- 用同一数据口径跑传统模型（Logistic Regression / Random Forest / Gradient Boosting）。
- 用统一指标评估：PR-AUC、F1、Recall、Recall@Precision、Brier、混淆矩阵。

为什么做：

- 给神经网络提供“强基线”，证明复杂模型的提升是有效而不是偶然。

如何运行：

```bash
scripts/run_in_pytorch_env.sh python -m src.models.baselines \
  --input data/interim/traverse_city_daytime_clean_v1.csv \
  --report artifacts/reports/baseline_results.json \
  --fig-dir artifacts/figures
```

输出：

- `artifacts/reports/baseline_results.json`
- `artifacts/figures/baseline_cm_*.png`

---

## 8. 模块 G：多模态神经网络训练（T20）

代码位置：

- `src/models/multimodal/dataset.py`
- `src/models/multimodal/model.py`
- `src/models/multimodal/train_loop.py`
- `src/models/multimodal/train.py`

各文件目的：

- `dataset.py`：把图像序列 + 气象序列对齐成可训练样本；支持图片内存缓存。
- `model.py`：ConvLSTM（图像分支）+ LSTM（气象分支）融合输出二分类 logit。
- `train_loop.py`：训练/验证循环，早停、LR 调度、指标计算。
- `train.py`：命令行训练入口，负责读数据、构建 DataLoader、保存 checkpoint 和 summary。

为什么做：

- 把 notebook 训练逻辑拆成可维护模块，支持本地可控调参和复现实验。

如何运行（默认推荐，自动选择设备）：

```bash
scripts/run_in_pytorch_env.sh python -m src.models.multimodal.train \
  --input-csv data/interim/traverse_city_daytime_clean_v1.csv \
  --image-dir data/processed/images/lake_michigan_64_png \
  --output-dir artifacts/reports \
  --checkpoint-path models/pytorch/michigancast_multimodal_best.pth \
  --device auto
```

Apple Silicon 推荐（已做 Metal + 内存自适应）：

```bash
scripts/run_in_pytorch_env.sh python -m src.models.multimodal.train \
  --device auto \
  --apple-metal-opt \
  --cache-images auto
```

输出：

- 模型权重：`models/pytorch/michigancast_multimodal_best.pth`
- 训练摘要：`artifacts/reports/multimodal_train_summary.json`
- 每轮指标表：`artifacts/reports/multimodal_epoch_metrics.csv`
- 训练曲线图：`artifacts/reports/multimodal_train_validation_metrics.png`

---

## 9. 从零到结果的推荐执行顺序

按这 7 步跑最稳：

1. `contracts.py`（先查数据契约）
2. `validate.py`（查时间/图像对齐）
3. `clean.py`（产出干净训练数据）
4. `eda_report.py`（看分布、季节性、漂移）
5. `baselines.py`（拿到传统 ML 基线）
6. `multimodal.train.py`（训练多模态模型）
7. 对比 `baseline_results.json` 和 `multimodal_train_summary.json`

---

## 10. 常见问题（你大概率会遇到）

问题 1：`[contracts] status=fail` 但程序退出码是 0。  
原因：当前契约脚本会输出失败报告，但不会主动抛异常退出。  
处理：看 `data_contract_report.json` 里的 `failed_checks` 具体列和行。

问题 2：训练时报 `Empty train/val dataset`。  
原因：`--nrows` 太小，或年份范围覆盖不到 val/test。  
处理：去掉 `--nrows`，或调整 `--train-years/--val-years/--test-years`。

问题 3：Apple 机器没走 MPS。  
原因：当前 PyTorch 构建可能不支持 MPS。  
处理：先用 `--device auto` 看日志中的 `device=...`，必要时升级 PyTorch。

---

## 11. 当前完成范围（方便对外说明）

已完成：

- 任务定义与防泄漏约束（T00-T02）
- 可靠数据与基线（T10-T15）
- 多模态模块化训练入口（T20）
- Apple Metal 与统一内存自适应训练优化（已集成）

未完成（后续可继续）：

- 系统化实验配置矩阵（T21）
- 训练 run 自动归档体系（T22）
- 不平衡策略专项实验（T23）
- 导出部署格式（T25）与服务化（T35）
