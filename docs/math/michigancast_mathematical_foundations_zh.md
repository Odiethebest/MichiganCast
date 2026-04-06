# MichiganCast 数学基础与建模决策说明

## 摘要

本文档给出 MichiganCast 中核心数学概念、公式与工程决策之间的对应关系。目标是让读者在不依赖实现细节的情况下，理解项目为何采用当前的数据构造、模型结构、损失函数、评估指标与稳定性标准。

---

## 1. 任务形式化与符号

设时间索引为 `t`，气象特征向量为 `m_t ∈ R^F`，卫星图像帧为 `I_t ∈ R^(1×H×W)`，降水实值为 `p_t`。

项目预测的是未来时刻是否降水，而非同刻分类。对预测步长 `h`（小时），定义：

$$
y_t^{(h)}=\mathbb{1}\!\left(p_{t+h}>\tau_{\text{rain}}\right), \quad \tau_{\text{rain}}=0
$$

输入窗口定义为：

$$
\mathbf{X}^{\text{meteo}}_t=\left[\mathbf{m}_{t-L_m+1},\ldots,\mathbf{m}_t\right],\quad
\mathbf{X}^{\text{img}}_t=\left[\mathbf{I}_{t-L_i+1},\ldots,\mathbf{I}_t\right]
$$

其中 `L_m` 与 `L_i` 分别是气象和图像回看长度。

---

## 2. 防泄漏约束与时间切分

### 2.1 防泄漏约束

项目在样本索引层强制：

$$
t < t+h
$$

并要求所有输入窗口结束索引严格早于标签索引（代码中对应 `validate_temporal_order`）。

### 2.2 时间切分

样本按标签时间 `y_time` 的年份切分为 train/val/test，而不是随机切分。形式上：

$$
\mathcal{D}_{\text{train}}=\{(x,y)\mid \text{year}(y\_time)\in [y_1,y_2]\}
$$

val/test 同理。该策略对应时序预测中的“未来不可见”原则。

---

## 3. 数据变换中的数学

### 3.1 标准化

在不平衡策略实验中，对数值特征做 z-score 标准化：

$$
\tilde{x}_{ij}=\frac{x_{ij}-\mu_j}{\sigma_j},\quad
\mu_j=\frac{1}{N}\sum_{i=1}^N x_{ij},\quad
\sigma_j=\sqrt{\frac{1}{N}\sum_{i=1}^N (x_{ij}-\mu_j)^2}
$$

并以 `max(sigma_j, 1e-6)` 避免除零。

### 3.2 缺失值插补与图像归一化

- 基线模型中，缺失值使用中位数插补：

$$
x_{ij}\leftarrow \operatorname{median}\{x_{kj}\}_{k\in \mathcal{I}_j}
$$

- 图像像素在数据集构造中被缩放到 `[-1, 1]`：

$$
u=\frac{\text{pixel}}{255},\quad \hat{u}=\frac{u-0.5}{0.5}
$$

---

## 4. 基线模型的数学形式

### 4.1 Logistic Regression

$$
P(y=1\mid \mathbf{x})=\sigma(\mathbf{w}^\top \mathbf{x}+b),\quad
\sigma(z)=\frac{1}{1+e^{-z}}
$$

训练最小化加权对数损失（含正则）。

### 4.2 Random Forest

随机森林是多棵树的集成，概率可写作：

$$
\hat{p}(\mathbf{x})=\frac{1}{B}\sum_{b=1}^{B} T_b(\mathbf{x})
$$

其中 `T_b` 为第 `b` 棵树的概率输出。

### 4.3 Gradient Boosting

加法模型形式：

$$
F_M(\mathbf{x})=\sum_{m=1}^M \nu \gamma_m h_m(\mathbf{x}),\quad
\hat{p}(\mathbf{x})=\sigma(F_M(\mathbf{x}))
$$

其中 `nu` 是学习率，`h_m` 是基学习器。

---

## 5. 多模态网络的核心公式

### 5.1 ConvLSTM 图像分支

对每个时刻 `t`，ConvLSTM 单元为：

$$
\mathbf{i}_t=\sigma(\mathbf{W}_{xi}*\mathbf{X}_t+\mathbf{W}_{hi}*\mathbf{H}_{t-1}+\mathbf{b}_i)
$$
$$
\mathbf{f}_t=\sigma(\mathbf{W}_{xf}*\mathbf{X}_t+\mathbf{W}_{hf}*\mathbf{H}_{t-1}+\mathbf{b}_f)
$$
$$
\mathbf{o}_t=\sigma(\mathbf{W}_{xo}*\mathbf{X}_t+\mathbf{W}_{ho}*\mathbf{H}_{t-1}+\mathbf{b}_o)
$$
$$
\tilde{\mathbf{C}}_t=\tanh(\mathbf{W}_{xg}*\mathbf{X}_t+\mathbf{W}_{hg}*\mathbf{H}_{t-1}+\mathbf{b}_g)
$$
$$
\mathbf{C}_t=\mathbf{f}_t\odot\mathbf{C}_{t-1}+\mathbf{i}_t\odot\tilde{\mathbf{C}}_t,\quad
\mathbf{H}_t=\mathbf{o}_t\odot\tanh(\mathbf{C}_t)
$$

其中 `*` 表示卷积，`⊙` 表示逐元素乘法。

### 5.2 LSTM 气象分支与融合头

气象序列经 LSTM 得到最终隐状态 `h_meteo`。图像分支最终特征记为 `h_img`（池化后向量）。

融合：

$$
\mathbf{h}^{\text{fusion}}=[\mathbf{h}^{\text{img}};\mathbf{h}^{\text{meteo}}]
$$
$$
z=\mathbf{W}_2\phi(\mathbf{W}_1\mathbf{h}^{\text{fusion}}+\mathbf{b}_1)+\mathbf{b}_2,\quad
\hat{p}=\sigma(z)
$$

其中 `z` 为 logit，`phi` 为 ReLU。

---

## 6. 损失函数与不平衡学习

### 6.1 BCEWithLogits

二分类交叉熵（以 logit `z` 表示）：

$$
\mathcal{L}_{\text{BCE}}(z,y)=
-y\log \sigma(z)-(1-y)\log(1-\sigma(z))
$$

### 6.2 加权 BCE

对正类加权 `w_+`：

$$
\mathcal{L}_{\text{WBCE}}(z,y)=
-w_+\,y\log \sigma(z)-(1-y)\log(1-\sigma(z))
$$

项目中 `w_+ ≈ N_neg / N_pos`（下限 1）。

### 6.3 Focal Loss

$$
p_t=
\begin{cases}
\hat{p}, & y=1\\
1-\hat{p}, & y=0
\end{cases},
\quad
\alpha_t=
\begin{cases}
\alpha, & y=1\\
1-\alpha, & y=0
\end{cases}
$$
$$
\mathcal{L}_{\text{focal}}=-\alpha_t(1-p_t)^\gamma\log(p_t)
$$

用于降低易分类样本权重，强调困难样本。

### 6.4 阈值移动（Threshold Moving）

给定精度下限 `pi_0`，项目在验证集上选择：

$$
\tau^*=\arg\max_{\tau}\ \text{Recall}(\tau)
\quad
\text{s.t.}\quad \text{Precision}(\tau)\ge \pi_0
$$

若无可行阈值，则退化为最大 recall 点。

---

## 7. 优化、正则与训练停止准则

### 7.1 Adam

令梯度为 `g_t = ∇_theta L_t`：

$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2
$$
$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\quad
\hat v_t=\frac{v_t}{1-\beta_2^t}
$$
$$
\theta_t=\theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

### 7.2 学习率调度与早停

- ReduceLROnPlateau：若验证损失长期不降，`eta <- 0.5 * eta`。
- Early Stopping：若验证损失连续 `P` 个 epoch 未改善，则停止训练。

---

## 8. 评估指标公式与决策含义

### 8.1 混淆矩阵与基础指标

$$
\text{Precision}=\frac{TP}{TP+FP},\quad
\text{Recall}=\frac{TP}{TP+FN}
$$
$$
F_1=2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}
$$

### 8.2 PR-AUC（Average Precision）

项目使用 average precision（AP）：

$$
\text{AP}=\sum_n (R_n-R_{n-1})P_n
$$

其中 `P_n, R_n` 分别是 PR 曲线上的 precision 与 recall 点。

### 8.3 Recall at Precision

$$
\text{Recall@Precision}(\pi_0)=
\max_{\tau:\ \text{Precision}(\tau)\ge \pi_0}\text{Recall}(\tau)
$$

### 8.4 Brier Score

$$
\text{Brier}=\frac{1}{N}\sum_{i=1}^{N}(\hat p_i-y_i)^2
$$

该项衡量概率预测校准误差。

---

## 9. 稳定性判据与服务监控统计

### 9.1 多次训练稳定性

对指标 `m_k`（如 test PR-AUC），在多次重复训练 `r = 1, ..., R` 下定义：

$$
\Delta_k=\max_r m_k^{(r)}-\min_r m_k^{(r)}
$$

再定义：

$$
\Delta_{\max}=\max_k \Delta_k
$$

若 `Delta_max <= delta`（项目中 `delta = 0.03`），则判定训练稳定。

### 9.2 在线监控统计

服务侧维护：

- 延迟均值、`p50`、`p95`、最大值；
- 预测分数的均值、标准差、最小/最大；
- 固定分箱直方图统计；
- 输入分布的均值与标准差摘要。

例如分位数定义：

$$
Q_p=\inf\{x: F(x)\ge p\}
$$

---

## 10. 公式到代码的映射

| 数学对象 | 代码位置 |
|---|---|
| 未来标签定义 `y_t^(h)` 与防泄漏约束 | `src/features/labeling.py` |
| 按标签年份切分 | `src/data/split.py` |
| 标准化与阈值移动 | `src/train/imbalance.py` |
| Logistic / RF / GB 基线 | `src/models/baselines.py` |
| ConvLSTM + LSTM 融合网络 | `src/models/multimodal/model.py` |
| BCE、Adam、调度器、早停 | `src/models/multimodal/train_loop.py` |
| 指标族（PR-AUC, F1, Recall@Precision, Brier） | `src/eval/metrics.py` |
| 稳定性 `Delta_max` 判据 | `src/train/train.py` |
| 在线统计（分位数、直方图、分布摘要） | `src/serve/monitoring.py` |

---

## 参考文献

[1] Hochreiter, S., & Schmidhuber, J. (1997). [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735). *Neural Computation*, 9(8), 1735-1780.

[2] Shi, X., Chen, Z., Wang, H., Yeung, D.-Y., Wong, W.-K., & Woo, W.-C. (2015). [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://arxiv.org/abs/1506.04214). *NeurIPS 2015*.

[3] Kingma, D. P., & Ba, J. (2015). [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). *ICLR 2015*.

[4] Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002). *ICCV 2017*.

[5] Breiman, L. (2001). [Random Forests](https://doi.org/10.1023/A:1010933404324). *Machine Learning*, 45, 5-32.

[6] Friedman, J. H. (2001). [Greedy Function Approximation: A Gradient Boosting Machine](https://doi.org/10.1214/aos/1013203451). *Annals of Statistics*, 29(5), 1189-1232.

[7] Brier, G. W. (1950). [Verification of Forecasts Expressed in Terms of Probability](https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2). *Monthly Weather Review*, 78(1), 1-3.

[8] Davis, J., & Goadrich, M. (2006). [The Relationship Between Precision-Recall and ROC Curves](https://dl.acm.org/doi/10.1145/1143844.1143874). *ICML 2006*.

[9] Saito, T., & Rehmsmeier, M. (2015). [The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets](https://doi.org/10.1371/journal.pone.0118432). *PLOS ONE*, 10(3):e0118432.

[10] Prechelt, L. (1998). [Early Stopping - But When?](https://doi.org/10.1007/3-540-49430-8_3). In *Neural Networks: Tricks of the Trade*. Springer.

[11] Bishop, C. M. (2006). [Pattern Recognition and Machine Learning](https://link.springer.com/book/10.1007/978-0-387-45528-0). Springer.

[12] Goodfellow, I., Bengio, Y., & Courville, A. (2016). [Deep Learning](https://www.deeplearningbook.org/). MIT Press.

[13] Hyndman, R. J., & Athanasopoulos, G. (2021). [Forecasting: Principles and Practice (3rd ed)](https://otexts.com/fpp3/). OTexts.
