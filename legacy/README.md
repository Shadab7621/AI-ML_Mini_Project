# 🤖 ML Classification Visualizer Pro

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An interactive, production-grade **Machine Learning Classification Visualizer** built with **Streamlit** and **Scikit-Learn**. 

This application lets you train, tune, diagnose, and benchmark **7 distinct classification algorithms** on the UCI Mushroom dataset in real-time, accompanied by publication-quality diagnostic plots, 5-fold cross-validation, and an automated multi-model leaderboard.

---

## 📑 Table of Contents

- [Key Features](#-key-features)
- [Classifiers Implemented](#-classifiers-implemented)
- [Evaluation Metrics & Diagnostics](#-evaluation-metrics--diagnostics)
- [Application Structure & Tabs](#-application-structure--tabs)
- [Dataset Details](#-dataset-details)
- [Prerequisites](#-prerequisites)
- [Installation & How to Run](#-installation--how-to-run)
  - [Option 1: Quick Start (Local)](#option-1-quick-start-local)
  - [Option 2: Using Virtual Environment (Recommended)](#option-2-using-virtual-environment-recommended)
- [Project Architecture](#-project-architecture)
- [Troubleshooting & FAQs](#-troubleshooting--faqs)

---

## ✨ Key Features

- **🧠 7 Machine Learning Algorithms**: From linear models to ensemble trees and distance-based classifiers.
- **⚙️ Dynamic Hyperparameter Tuning**: Adjust regularization (`C`), tree depth, kernels, pruning parameters, and learning rates interactively via the sidebar.
- **🛡️ Production ML Pipelines**: Built using `sklearn.pipeline.Pipeline` with `StandardScaler` automatically applied to distance/gradient-sensitive algorithms (SVM, Logistic Regression, KNN) to prevent data leakage.
- **📊 5 Diagnostic Visualizations**:
  - Seaborn Confusion Matrix Heatmap (counts + percentages)
  - ROC Curve with AUC & shaded integral
  - Precision-Recall Curve (PRC) with Average Precision (AP)
  - Feature Importance (Gini impurity for tree models, Permutation Importance for others)
  - Learning Curves (diagnose Underfitting vs Overfitting across dataset sizes)
- **🔁 5-Fold Stratified Cross-Validation**: Statistical validation with fold variance, min/max score distribution, and per-fold delta tables.
- **🏆 Automated Model Leaderboard**: Train and evaluate all 7 models side-by-side with a single click, complete with conditional heatmap styling and CSV export.
- **🎨 High-Contrast Professional UI**: Clean, accessible sidebar, custom metric cards, Google Inter typography, and violet brand accents.

---

## 🧠 Classifiers Implemented

| Classifier | Hyperparameters Exposed | Preprocessing Pipeline |
|---|---|---|
| **Support Vector Machine (SVM)** | `C`, `kernel` (rbf, linear, poly, sigmoid), `gamma` (scale, auto) | `StandardScaler` + `SVC(probability=True)` |
| **Logistic Regression** | `C`, `max_iter`, `solver` (lbfgs, saga, liblinear) | `StandardScaler` + `LogisticRegression` |
| **Random Forest** | `n_estimators`, `max_depth`, `min_samples_split`, `max_features` | Raw Encoded + `RandomForestClassifier` |
| **K-Nearest Neighbors (KNN)** | `n_neighbors (k)`, `weights`, `metric`, `algorithm` | `StandardScaler` + `KNeighborsClassifier` |
| **Decision Tree** | `max_depth`, `criterion` (gini, entropy, log_loss), `min_samples_split`, `min_samples_leaf`, `ccp_alpha` (Cost-Complexity Pruning) | Raw Encoded + `DecisionTreeClassifier` |
| **Gradient Boosting** | `n_estimators`, `learning_rate`, `max_depth`, `subsample` | Raw Encoded + `GradientBoostingClassifier` |
| **Gaussian Naive Bayes** | `var_smoothing` (log scale: $10^{-x}$) | Raw Encoded + `GaussianNB` |

---

## 📈 Evaluation Metrics & Diagnostics

The platform computes industry-standard evaluation metrics on the hold-out test set (30% split):

1. **Accuracy**: Overall fraction of correct predictions.
2. **Precision**: $\frac{TP}{TP + FP}$ — critical when false positives are costly.
3. **Recall / Sensitivity**: $\frac{TP}{TP + FN}$ — paramount for mushroom safety (identifying poisonous mushrooms).
4. **F1-Score**: Harmonic mean of Precision and Recall.
5. **ROC-AUC**: Area Under the Receiver Operating Characteristic Curve.
6. **Matthews Correlation Coefficient (MCC)**: High-fidelity metric balanced across all 4 confusion matrix quadrants (-1 to +1).
7. **Cohen’s Kappa ($\kappa$)**: Degree of agreement beyond chance.
8. **Full Classification Report**: Per-class precision, recall, f1, and support counts.

---

## 📑 Application Structure & Tabs

The app is organized into **5 dedicated analytical views**:

1. **📁 Tab 1 — Dataset Explorer**:
   - Target class distribution (Edible vs Poisonous counts and ratios).
   - Interactive raw dataframe viewer.
   - High-cardinality and feature metadata summary.
2. **🎯 Tab 2 — Model Results**:
   - 7 primary KPI metric cards with status colors (Green/Amber/Red).
   - Formatted Scikit-Learn Classification Report.
   - Predictions table with true vs predicted labels and one-click **CSV Download**.
3. **📈 Tab 3 — Diagnostic Plots**:
   - 3-column row displaying Confusion Matrix, ROC Curve, and PR Curve.
   - Top 15 Feature Importances bar chart.
   - Learning Curve showing training vs cross-validation convergence.
4. **🔁 Tab 4 — Cross-Validation**:
   - 5-Fold Stratified CV scores, mean accuracy, standard deviation, and fold range.
   - Fold-by-fold delta analysis chart and formatted breakdown table.
5. **🏆 Tab 5 — Compare All Models**:
   - Parallel training across all 7 algorithms.
   - Color-coded leaderboard table highlighting top performers.
   - Comparative multi-metric bar chart.
   - Exportable benchmark comparison CSV.

---

## 🍄 Dataset Details

- **Dataset**: UCI Mushroom Dataset (`Mushrooms.csv`)
- **Total Records**: 8,124 samples
- **Total Features**: 22 categorical attributes (cap-shape, odor, gill-color, habitat, etc.)
- **Target Variable**: `type` / `class`
  - `e` = Edible (51.8%)
  - `p` = Poisonous (48.2%)
- **Data Splitting**: Stratified 70% Train (5,686 samples) / 30% Test (2,438 samples), random state seeded at `42` for exact reproducibility.

---

## 💻 Prerequisites

Ensure you have Python installed on your machine:
- **Python 3.10, 3.11, or 3.12**
- **pip** (Python package installer)
- Git (optional, for cloning)

Verify installation:
```bash
python --version
pip --version
```

---

## 🚀 Installation & How to Run

### Option 1: Quick Start (Local)

1. Open your terminal or PowerShell and navigate to the project directory:
   ```bash
   cd "C:\Users\Sameer Khan\OneDrive\Desktop\Codestra\Classification-Algorithms-Visualiser"
   ```

2. Install all required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```
   *(Alternative command if `streamlit` is not in PATH)*:
   ```bash
   python -m streamlit run app.py
   ```

4. The app will automatically open in your default browser at:
   ```
   http://localhost:8501
   ```

---

### Option 2: Using Virtual Environment (Recommended)

Using a virtual environment prevents conflicts with existing Python packages:

#### On Windows (PowerShell / Command Prompt):
```powershell
# 1. Navigate to project directory
cd Classification-Algorithms-Visualiser

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
.\venv\Scripts\activate

# 4. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Run the web application
streamlit run app.py
```

#### On Linux / macOS:
```bash
# 1. Navigate to project directory
cd Classification-Algorithms-Visualiser

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run application
streamlit run app.py
```

---

## 🗂️ Project Architecture

```
Classification-Algorithms-Visualiser/
│
├── Mushrooms.csv          # UCI Mushroom dataset (8,124 rows, 22 features)
├── app.py                 # Core Streamlit application (pipeline, UI, models, plots)
├── requirements.txt       # Project dependencies and minimum versions
├── README.md              # Comprehensive project documentation
└── .gitignore             # Git ignore patterns for venv, cache, and OS files
```

---

## ❓ Troubleshooting & FAQs

### 1. `ImportError: cannot import name 'plot_confusion_matrix' from 'sklearn.metrics'`
> **Cause**: Scikit-Learn version 1.2+ deprecated and removed old plotting functions like `plot_confusion_matrix` and `plot_roc_curve`.  
> **Solution**: Already resolved in `app.py`! The application uses modern `ConfusionMatrixDisplay`, `RocCurveDisplay`, and `PrecisionRecallDisplay`.

### 2. Streamlit port already in use (`Address already in use: 8501`)
Run the app on an alternative port:
```bash
streamlit run app.py --server.port 8502
```

### 3. File Not Found Error (`Mushrooms.csv`)
Make sure your terminal's working directory is the project root containing `Mushrooms.csv` before executing `streamlit run app.py`.

### 4. Permutation Importance takes a few seconds
For non-tree models (like SVM or KNN), Scikit-Learn computes **Permutation Importance** by shuffling each of the 22 features across the test set. A loading spinner will indicate calculation progress.

---

## 📜 License

Distributed under the MIT License. Feel free to use, modify, and extend this project for academic or commercial purposes.
