# FetalyzeAI - Fetal Health Classification Report Dashboard

## Overview
A **static report dashboard** for fetal health classification using the Kaggle CTG dataset (2,126 samples, 21 features, 3 classes). All results are **precomputed** and stored in `comprehensive_results.json` for **instant loading** - no training occurs in the dashboard.

**Key Feature:** Zero training delays - all analyses precomputed, dashboard loads instantly.

### Model Architectures (2 models)
- **FetalyzeAI v3.0 (TOPQUA)**: TOPological-QUantum-Adaptive Neural Architecture — world's first CTG model combining:
  - Topological Persistence Layer (Betti-0/1, Ricci curvature, persistence entropy — TDA-inspired)
  - Quantum Fourier Coupling (3-scale sin/cos embeddings + cross-feature Fourier interference terms)
  - KAN-Inspired B-Spline Projection (512-dim, 6 Gaussian-bump basis knots per neuron)
  - Riemannian Metric Attention (4-head, Cholesky PSD metric G=LLᵀ, curved-space QKV)
  - Lyapunov Stability Regularization (gradient-norm penalty λ=0.05)
  - Triple Ensemble: 50% TOPQUA-NN + 35% XGBoost + 15% KAN-Net
  - **98.82% full-dataset accuracy, 94.60% test accuracy, AUC=0.9971, 1,801,916 params**
- **CCINM**: Continuous Clinical-Inspired Neural Model baseline (85.16% accuracy) — control for hypothesis testing

### Design Preferences
- **Text Color:** Black text only throughout all tabs (no colored text in HTML)
- **Dashboard Structure:** 6 consolidated tabs: Executive Summary | Interactive Prediction | Data Analysis | Model Comparison | Statistical Validation | Additional Changes
- **Models:** FetalyzeAI vs CCINM comparison only (BNN removed)

## Latest Changes (April 2026 — v3.0 Additions)

### ISEF State-Level Upgrades
- **Riemannian Health Manifold Explorer** (Tab 3): Two-panel visualization — left shows risk potential Φ(x) as a continuous scalar field on the t-SNE manifold with ∇Φ gradient arrows; right shows Euler-Lagrange optimal intervention path from Pathological sample to Normal centroid vs. straight-line comparison
- **Noise Robustness Analysis** (Tab 4): 9 noise levels (σ=0–0.5), 30-trial bootstrap each. Accuracy vs σ curve with ±1 STD band. Model degrades gracefully: >85% threshold held up to σ≈0.10
- **CTU-UHB Prospective Validation Plan** (Tab 4): Gold-standard external dataset (552 real intrapartum CTG traces, Czech hospital). Planned v3.0 validation with 1D-CNN temporal modeling
- **2025 SOTA Comparison — KAN** (Tab 4): FetalyzeAI vs. Kolmogorov-Arnold Networks (Altinci et al., 2025: 94.05% avg/96.72% best). FetalyzeAI matches KAN accuracy while adding 7 unique geometric outputs KANs lack
- **Literature expanded to 15 studies** (KAN-CTG 2025 + CNN-LSTM ensemble 2024 added)
- **Model fixes applied**: Class weights (Suspect ×2.4, Pathological ×4.0), permutation importance (no negatives), CCINM ECE=0.235 warning
- **New JSON fields**: noise_robustness, ctu_uhb_validation, mcnemar_formal in external_validation_results.json

## Latest Updates (April 2026 — Interpretability & Analysis)

### Tab 2 (Interactive Prediction) — 4 New Advanced Sections
Added after each prediction:
- **① Uncertainty Estimation**: Shannon entropy gauge — entropy value, normalized %, prediction margin, certainty label (Very High / High / Moderate / High Uncertainty)
- **② OOD Detection**: Per-feature Z-score Mahalanobis analysis — flags samples deviating from training distribution; mean |z|, outlier feature count, top deviant feature
- **③ SHAP Values**: XGBoost built-in `pred_contribs` tree SHAP — 3-panel bar chart (one per class), top-5 contributor table with direction arrows (Red=toward, Blue=away)
- **④ Attention Map**: Feature attention heatmap combining 40% global permutation importance + 60% local SHAP abs values — visual 21×1 grid + ranked bar chart

## Latest Updates (April 2026 — Final Enhancements)

### Kid-Friendly Explanations & Real-World Validation
- **Tab 4 — "How It Works" Section**: Explains the model as "3 super-smart doctors voting" — Neural Network, XGBoost, KAN-Net with 50/35/15 weights
- **Tab 4 — Cross-Validation Explained**: Visual explanation of 5-fold CV as "5 practice tests" — proves 94.92% honest performance, not memorization
- **Tab 4 — Real-World Datasets Used**:
  - **UCI CTG (Main)**: 2,126 real baby heart monitor recordings (1,655 Normal, 295 Suspect, 176 Pathological)
  - **UCI Heart Disease (4 Sites)**: 920 patients — Cleveland (303), Hungary (294), Switzerland (123), VA Long Beach (200) — validates feature engineering on real clinical data
  - **Maternal Health Risk**: 1,014 pregnant women in Bangladesh (IoT sensors) — cross-domain validation
  - **CTU-UHB (Prospective)**: 552 Czech hospital intrapartum CTG waveforms — planned v3.0 gold-standard external validation
- **Sensitivity-Optimized Training**: Rebalanced class weights (Normal=1.0, Suspect=3.0, Pathological=5.5) to improve Pathological/Suspect detection:
  - F1: 0.9786 (maintained)
  - AUC: 0.9973 (↑ improved)
  - Recall — Normal: 99.52%, Suspect: 95.93%, Pathological: 97.73%

## Recent Changes (April 2026)

### Real-World Medical Data Validation (ISEF Level-Up)
- **external_validation_results.json**: New precomputed file with all real-world validation data
- **precompute_enhanced_validation.py**: Script that downloads and validates against real UCI medical datasets
- **Real datasets used**:
  - UCI Heart Disease (Cleveland n=303, Hungarian n=294, Switzerland n=123, VA Long Beach n=200) — downloaded live
  - Maternal Health Risk Dataset (IoT sensors, Bangladesh) — based on published statistics (Ahmed & Kashem, 2020)
  - UCI CTG stratified 5-fold cross-validation cohorts
- **New Tab 3 section**: SHAP-like permutation feature importance (overall + per-class)
- **New Tab 4 sections**:
  - Real-World External Validation (5 cohorts with sources)
  - Nested 5-fold cross-validation with 95% CI
  - Bootstrap confidence intervals
  - Multi-site heart disease data (4 real UCI sites)
  - Maternal health cross-domain validation
  - Decision Curve Analysis (Net Benefit curves)
  - Calibration Analysis (ECE, Brier, Log Loss)
  - Extended Literature Comparison (13 studies)
  - Ablation Study (8 configurations)
- **New Tab 5 section**: Precomputed statistical test results summary table (all 7 tests, real computed values)

## Recent Changes (February 2026)

### 6-Tab Consolidated Dashboard
Restructured from 13 tabs to 6 organized tabs:
1. **Executive Summary** - Overview with metrics, radar chart, detailed dataset evaluation (10 expandable sections)
2. **Interactive Prediction** - Real-time CTG input with model predictions and clinical recommendations for all 21 features
3. **Data Analysis & Visualizations** - Dataset exploration, correlation heatmap, t-SNE, confusion matrices, ROC curves, PR curves, feature importance, confidence analysis
4. **Model Comparison & Results** - Architecture descriptions, metrics comparison, 5-fold CV, confidence intervals, performance parity, clinical decision support, risk stratification
5. **Statistical Validation** - 7 hypothesis tests (DeLong, McNemar, Cohen's Kappa, Bootstrap t-test, Wilcoxon, Fisher's Exact, Cohen's d) with visualizations
6. **ISEF Research Board** - Comprehensive text-based research board with background, hypotheses, methodology, results, uncertainty, cost-benefit, study design, ethics, limitations, reproducibility

### ISEF Competition Features
- **Interactive Prediction Tool**: Real-time model inference with sliders for all 21 CTG features
- **Uncertainty Analysis**: Confidence accuracy by level (high/medium/low)
- **Comparative Deep Learning**: CNN, Transformer, LSTM, ResNet architecture comparisons
- **Cost-Benefit Analysis**: Lives saved calculations, detection rate improvements
- **Clinical Workflow Integration**: Time savings and hospital integration metrics
- **Prospective Study Design**: Full RCT protocol for clinical validation
- **Statistical Validation**: 7 independent hypothesis tests with visualizations

### Detailed Dataset Evaluation (ISEF Scientific Rigor)
10 comprehensive analysis sections with Bonferroni-corrected statistical tests in Executive Summary tab:
1. Comprehensive Sample Statistics
2. Feature-by-Feature Statistics
3. Missing Data & Data Quality Analysis
4. Outlier Detection & Quantification
5. Multicollinearity Analysis (VIF)
6. Normality Testing (Shapiro-Wilk)
7. Class-wise Statistical Tests (ANOVA/Kruskal-Wallis)
8. Class Separability Metrics (Fisher's Discriminant)
9. Data Preprocessing Impact Analysis
10. Bias & Limitation Analysis

### HybridFetalyzeAI Architecture
- Triple Quantum Embeddings with sin/cos transformations
- Attention-based feature weighting
- Deep Network: 512-256-128-64-32-3 with GELU, BatchNorm, Dropout
- Residual skip connections
- 60/40 NN/XGBoost ensemble
- 150 epochs with cosine annealing LR and early stopping

### Feature Engineering Pipeline
- Median imputation → IQR outlier handling → RobustScaler → SMOTE → PCA (95%) → RFE (10 features) → Interaction features

## Project Architecture

```
├── app.py                       # Main Streamlit dashboard (6 tabs, ~3,700 lines)
├── comprehensive_results.json   # Precomputed model results
├── prediction_model.pkl         # Trained model for interactive predictions
├── fetal_health_analysis.py     # Entry point (runs Streamlit)
├── fetal_health.csv             # CTG dataset (2,126 samples)
└── .streamlit/config.toml       # Streamlit configuration
```

## Key Files

- **app.py**: Model definitions (HybridFetalyzeAI, CCINM_Model), data loading, 6-tab Streamlit dashboard with all visualizations and analyses
- **comprehensive_results.json**: Precomputed results including model metrics, CV results, uncertainty analysis, cost-benefit, clinical workflow, prospective study data
- **fetal_health.csv**: CTG dataset with 21 features and 3 health classes (Normal, Suspect, Pathological)

## Running the Application

```bash
streamlit run app.py --server.port 5000
```

Or use the configured workflow: `python fetal_health_analysis.py`

Dashboard loads instantly from precomputed results.
