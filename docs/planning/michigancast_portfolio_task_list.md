# MichiganCast 项目任务清单（Portfolio 升级版）

目标：把当前课设升级为“可复现、可部署、可评估”的多模态降水预测项目，并明确展示以下三项能力：
1. 数据清洗、数据分析、机器学习能力
2. 人工神经网络手动调参与本地训练能力
3. 数据工程能力

---

## 0. 统一问题定义（先做）

| ID | 任务 | 能力映射 | 产出 | 完成标准 |
|---|---|---|---|---|
| T00 | 重新定义预测目标：从“同刻判别”改为“未来预测（t+6h / t+24h）” | 1,2 | `configs/task.yaml` | 明确输入窗口、预测步长、标签时刻，且训练特征不包含目标时刻信息 |
| T01 | 修复标签泄漏风险（禁止使用与标签同刻的 `Precip_in` 信息） | 1,2 | `src/features/labeling.py` | 代码层面保证 `X_time < y_time`，并有自动校验 |
| T02 | 切分策略改为时间切分（train/val/test 按年份） | 1,3 | `src/data/split.py` | 不再使用随机切分；可打印每段时间范围与样本数 |

---

## 1. 数据清洗 + 数据分析 + 机器学习（能力 1 主线）

| ID | 任务 | 能力映射 | 产出 | 完成标准 |
|---|---|---|---|---|
| T10 | 建立数据契约（字段类型、缺失值规则、取值范围） | 1,3 | `src/data/contracts.py` | 运行后输出校验报告；不合格数据可定位到行 |
| T11 | 数据质量检查：缺失码、坏图像、时间连续性、图像-表格对齐 | 1,3 | `src/data/validate.py` | 生成 `artifacts/data_quality_report.json` |
| T12 | 清洗流水线脚本化（替代 notebook 手工步骤） | 1,3 | `src/data/clean.py` | 一条命令生成干净数据集，结果可复现 |
| T13 | 构建 EDA 自动报告（类别分布、季节性、特征相关、漂移） | 1 | `src/analysis/eda_report.py` | 输出图表与关键统计，含类别不平衡结论 |
| T14 | 建传统基线模型（Logistic + 树模型至少 2 个） | 1 | `src/models/baselines.py` | 给出 PR-AUC、F1、Recall、混淆矩阵 |
| T15 | 统一评估口径（时间切分 + 稀有事件指标） | 1 | `src/eval/metrics.py` | 指标包含 PR-AUC、F1、Recall@Precision、Brier |

---

## 2. 神经网络手动调参 + 本地训练（能力 2 主线）

| ID | 任务 | 能力映射 | 产出 | 完成标准 |
|---|---|---|---|---|
| T20 | 将多模态模型代码模块化（Dataset / Model / Train Loop） | 2,3 | `src/models/multimodal/` | 训练入口脱离 notebook，可 CLI 启动 |
| T21 | 定义“手动调参矩阵”（窗口、hidden、dropout、lr、loss、阈值） | 2 | `configs/experiments/*.yaml` | 至少 10 组人工设计实验配置 |
| T22 | 本地训练记录系统（每次实验自动记录配置和指标） | 2,3 | `artifacts/experiments/` | 每次 run 都有独立目录和 `metrics.json` |
| T23 | 不平衡学习策略实验（class weight / focal loss / threshold moving） | 2 | `src/train/imbalance.py` | 对比表中至少 1 项正类指标显著提升 |
| T24 | 训练稳定性增强（EarlyStopping、LR Scheduler、Seed 固定） | 2 | `src/train/train.py` | 连续两次同配置结果波动在可接受区间 |
| T25 | 模型导出（TorchScript/ONNX 至少一种） | 2,3 | `artifacts/models/` | 导出模型可被独立推理脚本加载 |

---

## 3. 数据工程与可交付能力（能力 3 主线）

| ID | 任务 | 能力映射 | 产出 | 完成标准 |
|---|---|---|---|---|
| T30 | 目录重构：`src/`, `configs/`, `data/`, `artifacts/`, `scripts/` | 3 | 新目录结构 | 根目录无核心逻辑散落在 notebook |
| T31 | 数据分层（raw / processed / features）与版本化（DVC 或等价方案） | 3 | `data/README` + 版本配置 | 任意版本可回溯来源和生成步骤 |
| T32 | 大文件优化：CSV 转 Parquet + 索引映射 | 3 | `src/data/build_parquet.py` | 读取速度明显提升，内存占用可控 |
| T33 | 一键流水线命令（`make` 或 `python -m`） | 3 | `Makefile` 或 `src/cli.py` | 可从校验到训练到评估一键执行 |
| T34 | 自动化测试（数据、特征、训练 smoke test） | 3 | `tests/` | CI 能跑通最小样本测试 |
| T35 | 轻量推理服务（FastAPI） | 2,3 | `src/serve/app.py` | 本地可启动 API 并返回降水概率 |
| T36 | 运行监控基础（输入统计、预测分布、延迟） | 3 | `src/serve/monitoring.py` | 每次推理可产生日志并可追踪 |

---

## 4. 执行顺序（建议）

1. 先完成 T00-T02（防止后面都基于错误任务定义）
2. 再完成 T10-T15（建立可靠数据与基线）
3. 接着完成 T20-T25（多模态模型与调参闭环）
4. 最后完成 T30-T36（工程化与可交付）

---

## 5. Portfolio 最小可展示版本（MVP）

以下 12 项完成即可对外展示工程能力闭环：
`T00 T01 T02 T10 T11 T12 T14 T15 T21 T22 T33 T35`
