# MichiganCast Notebook Refactor Plan

## 1. 背景与问题

当前 notebook 主要是课设开发过程记录，不适合直接用于 demo 展示。核心问题：

1. 内容重叠：`02_preprocessing` 与 `03_training` 有大量重复清洗/特征工程片段。
2. 体量过大：`03_training` 单本过长，不利于演示和面试讲解。
3. 工程口径不一致：仍有历史硬编码路径、旧流程描述，与当前 `src/` 模块化实现脱节。
4. 可读性不稳定：中英文混排、说明粒度不一致，叙事主线不清晰。
5. 安全与专业形象风险：存在不应公开保留的旧环境信息描述。

---

## 2. 重构目标

将 notebook 从“过程记录”改为“可演示文档层”，满足以下目标：

1. 每本 notebook 只讲一个主题问题，避免重复。
2. 所有关键步骤优先调用 `src/` 代码，不再在 notebook 重写业务逻辑。
3. 标题层级统一，便于现场讲解。
4. 内容与任务清单 T00-T36 对齐，能够映射工程能力。
5. 形成从数据到服务的完整 demo 闭环。

---

## 3. 目标目录结构

建议结构：

```text
notebooks/
├── demo/
│   ├── 00_demo_index.ipynb
│   ├── 01_data_foundation.ipynb
│   ├── 02_analysis_and_baselines.ipynb
│   ├── 03_multimodal_training_and_tuning.ipynb
│   └── 04_imbalance_stability_export_serve.ipynb
└── legacy/
    ├── 00_util/read_large_weather_csv_experiments.ipynb
    ├── 01_eda/lake_effect_precipitation_exploration.ipynb
    ├── 02_preprocessing/lake_michigan_satellite_preprocessing.ipynb
    └── 03_training/multimodal_rainfall_training_pipeline.ipynb
```

说明：

- `demo/`：对外展示用，章节规范、流程可复现。
- `legacy/`：历史材料归档，不参与主叙事。

---

## 4. Demo Notebook 拆分设计

### 4.1 `00_demo_index.ipynb`

定位：总览页（导航 + 能力映射）

内容：

1. 项目目标与预测任务定义（t+6h / t+24h）
2. 端到端流程图（data -> train -> export -> serve）
3. 每本 notebook 的阅读顺序与产出清单
4. 与任务清单 T00-T36 对应关系表

---

### 4.2 `01_data_foundation.ipynb`

定位：数据工程与数据质量基础能力

对应任务：T10, T11, T12, T30, T31

内容：

1. 数据层定义（raw/interim/processed/features/reference）
2. 数据契约校验（`src.data.contracts`）
3. 数据质量检查（`src.data.validate`）
4. 清洗流水线（`src.data.clean`）
5. 数据版本 manifest（`src.data.versioning`）

输出重点：

- `artifacts/reports/data_contract_report*.json`
- `artifacts/data_quality_report.json`
- `artifacts/reports/data_cleaning_summary.json`
- `configs/data/versions/*.json`

---

### 4.3 `02_analysis_and_baselines.ipynb`

定位：数据分析 + 传统机器学习能力

对应任务：T13, T14, T15

内容：

1. EDA 自动报告（`src.analysis.eda_report`）
2. 类别不平衡解释与指标选择依据
3. 基线训练（`src.models.baselines`）
4. 指标读取与结果对比（PR-AUC/F1/Recall/Brier）

输出重点：

- `artifacts/reports/eda_summary.json`
- `artifacts/reports/baseline_results.json`
- `artifacts/figures/*`

---

### 4.4 `03_multimodal_training_and_tuning.ipynb`

定位：神经网络建模 + 手动调参与本地训练

对应任务：T20, T21, T22

内容：

1. 多模态架构简述（ConvLSTM + LSTM）
2. 调参矩阵读取（`configs/experiments/t21_exp*.yaml`）
3. 训练入口调用（`src.models.multimodal.train`）
4. 每轮曲线与实验 run 记录展示

输出重点：

- `artifacts/reports/multimodal_train_summary.json`
- `artifacts/reports/multimodal_epoch_metrics.csv`
- `artifacts/reports/multimodal_train_validation_metrics.png`
- `artifacts/experiments/<run_id>/metrics.json`

---

### 4.5 `04_imbalance_stability_export_serve.ipynb`

定位：生产化能力闭环

对应任务：T23, T24, T25, T35, T36

内容：

1. 不平衡策略实验（`src.train.imbalance`）
2. 稳定性训练双跑（`src.train.train`）
3. 模型导出 TorchScript（`src.train.export`）
4. 独立推理脚本（`src.serve.infer_torchscript`）
5. FastAPI 服务 + 监控摘要（`src.serve.app` / `src.serve.monitoring`）

输出重点：

- `artifacts/reports/imbalance_strategy_comparison.*`
- `artifacts/reports/stability_train_report*.json`
- `artifacts/models/*.ts`
- `/metrics/summary` 返回示例

---

## 5. 统一标题模板（每本 notebook）

每本 notebook 使用同一结构：

1. `# MichiganCast Demo XX — <主题>`
2. `## 0. 本章目标与结论`
3. `## 1. 输入与输出（路径与产物）`
4. `## 2. 方法与实现（调用 src 模块）`
5. `## 3. 结果解读（图表与指标）`
6. `## 4. 工程反思与下一步`
7. `## Appendix. 复现实验命令`

规范：

- Markdown 解释优先于长代码块。
- 每段代码前说明“为什么执行、预期产物是什么”。
- 不在 notebook 中复制核心业务逻辑，统一调用 `python -m src....`

---

## 6. 旧 Notebook 到新结构映射

| 旧文件 | 主要可复用内容 | 迁移目标 |
|---|---|---|
| `00_util/read_large_weather_csv_experiments.ipynb` | CSV 读取尝试思路 | 仅少量迁入 `01_data_foundation`，其余归档 |
| `01_eda/lake_effect_precipitation_exploration.ipynb` | 可视化讲解文本 | 迁入 `02_analysis_and_baselines` |
| `02_preprocessing/lake_michigan_satellite_preprocessing.ipynb` | 图像处理与数据清洗思路 | 拆分迁入 `01` 与 `03` |
| `03_training/multimodal_rainfall_training_pipeline.ipynb` | 训练流程主线 | 拆分迁入 `03` 与 `04` |

---

## 7. 实施顺序（建议）

1. 新建 `notebooks/demo/00` 和 `01`，先定风格与模板。
2. 再完成 `02`（EDA+baseline）与 `03`（训练+调参）。
3. 最后完成 `04`（不平衡+稳定性+导出+服务）。
4. 将旧 notebook 移入 `notebooks/legacy/`。
5. 在 README 与 run 文档中更新 notebook 导航入口。

---

## 8. 验收标准

完成后应满足：

1. `demo/` 下 5 本 notebook 均可顺序执行。
2. 每本有统一大标题/小标题结构和清晰叙事。
3. 每本都能对应到明确的工程任务与产出文件。
4. 演示中不需要解释历史 notebook 的重复与偏差内容。

