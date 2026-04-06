# Manual Hyperparameter Matrix (T21)

This folder contains manually designed experiment configurations for multimodal forecasting.

Coverage dimensions:

- Window sizes: `meteo_lookback_steps`, `image_lookback_steps`
- Hidden sizes: `conv_hidden_dim`, `meteo_hidden_dim`, `fusion_hidden_dim`
- Regularization: `dropout`
- Optimizer: `learning_rate`, `weight_decay`
- Loss choice: `bce`, `weighted_bce`, `focal`
- Decision threshold: `decision_threshold`

Config files:

- `t21_exp01_baseline.yaml`
- `t21_exp02_short_window_fast.yaml`
- `t21_exp03_long_window_context.yaml`
- `t21_exp04_large_hidden.yaml`
- `t21_exp05_small_hidden.yaml`
- `t21_exp06_high_dropout.yaml`
- `t21_exp07_low_lr_stable.yaml`
- `t21_exp08_very_low_lr.yaml`
- `t21_exp09_focal_loss.yaml`
- `t21_exp10_weighted_bce.yaml`
- `t21_exp11_horizon6_fast_response.yaml`
- `t21_exp12_low_threshold_recall.yaml`
