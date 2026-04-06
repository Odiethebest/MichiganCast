# MichiganCast Mathematical Foundations and Modeling Decisions

## Abstract

This document summarizes the mathematical concepts and formulas used in MichiganCast, and explains how they connect to engineering decisions in the codebase. The goal is to provide enough theory for readers to understand the design of data construction, model architecture, loss functions, evaluation metrics, and stability criteria.

---

## 1. Problem Formulation and Notation

Let time index be `t`, meteorological feature vector be `m_t ∈ R^F`, satellite frame be `I_t ∈ R^(1×H×W)`, and precipitation value be `p_t`.

The project predicts future rain events, not same-time classification. For forecast horizon `h` (hours):

$$
y_t^{(h)}=\mathbb{1}\!\left(p_{t+h}>\tau_{\text{rain}}\right), \quad \tau_{\text{rain}}=0
$$

Input windows:

$$
\mathbf{X}^{\text{meteo}}_t=\left[\mathbf{m}_{t-L_m+1},\ldots,\mathbf{m}_t\right],\quad
\mathbf{X}^{\text{img}}_t=\left[\mathbf{I}_{t-L_i+1},\ldots,\mathbf{I}_t\right]
$$

where `L_m` and `L_i` are lookback lengths for meteorological and image sequences.

---

## 2. Leakage Constraints and Time Split

### 2.1 Leakage Constraint

At sample-index level, MichiganCast enforces:

$$
t < t+h
$$

and all input windows must end strictly before the target index (`validate_temporal_order` in code).

### 2.2 Time-based Split

Samples are split by target timestamp year (`y_time`), not by random shuffle:

$$
\mathcal{D}_{\text{train}}=\{(x,y)\mid \text{year}(y\_time)\in [y_1,y_2]\}
$$

Validation and test are defined analogously. This follows the temporal non-leakage principle for forecasting.

---

## 3. Mathematical Elements in Data Transformation

### 3.1 Standardization

In imbalance experiments, numerical features are z-score standardized:

$$
\tilde{x}_{ij}=\frac{x_{ij}-\mu_j}{\sigma_j},\quad
\mu_j=\frac{1}{N}\sum_{i=1}^N x_{ij},\quad
\sigma_j=\sqrt{\frac{1}{N}\sum_{i=1}^N (x_{ij}-\mu_j)^2}
$$

with `max(sigma_j, 1e-6)` for numerical safety.

### 3.2 Imputation and Image Normalization

- Median imputation for baseline models:

$$
x_{ij}\leftarrow \operatorname{median}\{x_{kj}\}_{k\in \mathcal{I}_j}
$$

- Pixel normalization to `[-1, 1]`:

$$
u=\frac{\text{pixel}}{255},\quad \hat{u}=\frac{u-0.5}{0.5}
$$

---

## 4. Baseline Model Mathematics

### 4.1 Logistic Regression

$$
P(y=1\mid \mathbf{x})=\sigma(\mathbf{w}^\top \mathbf{x}+b),\quad
\sigma(z)=\frac{1}{1+e^{-z}}
$$

The objective is weighted log loss with regularization.

### 4.2 Random Forest

Probability prediction from tree ensemble:

$$
\hat{p}(\mathbf{x})=\frac{1}{B}\sum_{b=1}^{B} T_b(\mathbf{x})
$$

where `T_b` is the `b`-th tree output.

### 4.3 Gradient Boosting

Additive boosting model:

$$
F_M(\mathbf{x})=\sum_{m=1}^M \nu \gamma_m h_m(\mathbf{x}),\quad
\hat{p}(\mathbf{x})=\sigma(F_M(\mathbf{x}))
$$

with learning rate `nu`, weak learner `h_m`, and step size `gamma_m`.

---

## 5. Core Equations of the Multimodal Network

### 5.1 ConvLSTM Image Branch

For each step `t`, ConvLSTM cell:

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

`*` is convolution, `⊙` is elementwise multiplication.

### 5.2 Meteorological LSTM Branch and Fusion Head

Let `h_meteo` be the final hidden state of the meteorological LSTM and `h_img` be the pooled image feature.

$$
\mathbf{h}^{\text{fusion}}=[\mathbf{h}^{\text{img}};\mathbf{h}^{\text{meteo}}]
$$
$$
z=\mathbf{W}_2\phi(\mathbf{W}_1\mathbf{h}^{\text{fusion}}+\mathbf{b}_1)+\mathbf{b}_2,\quad
\hat{p}=\sigma(z)
$$

where `z` is the logit and `phi` is ReLU.

---

## 6. Loss Functions and Imbalance Handling

### 6.1 BCE with Logits

$$
\mathcal{L}_{\text{BCE}}(z,y)=
-y\log \sigma(z)-(1-y)\log(1-\sigma(z))
$$

### 6.2 Weighted BCE

Positive class weighting `w_+`:

$$
\mathcal{L}_{\text{WBCE}}(z,y)=
-w_+\,y\log \sigma(z)-(1-y)\log(1-\sigma(z))
$$

In code, `w_+ ≈ N_neg / N_pos` (lower-bounded by 1).

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

### 6.4 Threshold Moving

Given minimum precision `pi_0`, choose threshold on validation set:

$$
\tau^*=\arg\max_{\tau}\ \text{Recall}(\tau)
\quad
\text{s.t.}\quad \text{Precision}(\tau)\ge \pi_0
$$

If infeasible, fallback to threshold with maximal recall.

---

## 7. Optimization, Regularization, and Stopping Criteria

### 7.1 Adam

For gradient `g_t = ∇_theta L_t`:

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

### 7.2 LR Scheduler and Early Stopping

- ReduceLROnPlateau: if validation loss stagnates, `eta <- 0.5 * eta`.
- Early stopping: stop if validation loss does not improve for `P` epochs.

---

## 8. Evaluation Metrics and Decision Semantics

### 8.1 Confusion Matrix Derived Metrics

$$
\text{Precision}=\frac{TP}{TP+FP},\quad
\text{Recall}=\frac{TP}{TP+FN}
$$
$$
F_1=2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}
$$

### 8.2 PR-AUC as Average Precision

$$
\text{AP}=\sum_n (R_n-R_{n-1})P_n
$$

with `P_n` and `R_n` from precision-recall curve points.

### 8.3 Recall at Precision Threshold

$$
\text{Recall@Precision}(\pi_0)=
\max_{\tau:\ \text{Precision}(\tau)\ge \pi_0}\text{Recall}(\tau)
$$

### 8.4 Brier Score

$$
\text{Brier}=\frac{1}{N}\sum_{i=1}^{N}(\hat p_i-y_i)^2
$$

---

## 9. Stability Criterion and Monitoring Statistics

### 9.1 Repeated-Run Stability

For metric `m_k` over runs `r = 1, ..., R`:

$$
\Delta_k=\max_r m_k^{(r)}-\min_r m_k^{(r)}
$$
$$
\Delta_{\max}=\max_k \Delta_k
$$

Training is considered stable when `Delta_max <= delta` (project default `delta = 0.03`).

### 9.2 Online Monitoring Summaries

Serving monitor tracks:

- latency mean, `p50`, `p95`, max
- prediction mean, std, min, max
- fixed-bin histogram of prediction scores
- input distribution summaries

Quantile definition:

$$
Q_p=\inf\{x: F(x)\ge p\}
$$

---

## 10. Formula to Code Mapping

| Mathematical Object | Code |
|---|---|
| Future label and anti-leakage constraints | `src/features/labeling.py` |
| Label-year time split | `src/data/split.py` |
| Standardization and threshold moving | `src/train/imbalance.py` |
| Logistic RF GB baselines | `src/models/baselines.py` |
| ConvLSTM plus LSTM fusion model | `src/models/multimodal/model.py` |
| BCE Adam scheduler early stopping | `src/models/multimodal/train_loop.py` |
| Metrics (PR-AUC F1 Recall@Precision Brier) | `src/eval/metrics.py` |
| Stability criterion `Delta_max` | `src/train/train.py` |
| Monitoring statistics and histogram | `src/serve/monitoring.py` |

---

## References

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
