# Breast Cancer Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange)
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-lightgrey)
![Plotly](https://img.shields.io/badge/Plotly-5.0%2B-green)

A machine learning project to classify breast tumors as **malignant (cancerous)** or **benign (non-cancerous)** using clinical features. Achieves **>97% accuracy** with optimized models.

## 📌 Project Overview
Breast cancer is the most common cancer among women worldwide. Early detection significantly improves survival rates. This project:
- Preprocesses clinical tumor data (handling outliers, scaling, feature selection).
- Trains and compares **5 ML models** (Decision Tree, KNN, Naïve Bayes, MLP, SVM).
- Identifies key diagnostic features and evaluates model performance.

## 📂 Dataset
**Source**: [Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))  
**Features**:  
- 30 numerical features (e.g., `radius_mean`, `texture_mean`, `smoothness_se`)  
- Target: `diagnosis` (Malignant=1, Benign=0)  

## 🛠️ Workflow
### 1. Data Preprocessing
- **Handled Missing Values**
- **Feature Engineering**:
  - Encoded target: `M` → 1 (Malignant), `B` → 0 (Benign).
  - Standardized features using `StandardScaler`.
- **Outlier Treatment**: Replaced outliers with KNN-imputed values.
- **Feature Selection**: Dropped highly correlated features (threshold=0.8).

### 2. Model Training & Evaluation
Trained 5 models using **GridSearchCV** for hyperparameter tuning:

| Model               | Best Parameters                              | Test Accuracy |
|---------------------|---------------------------------------------|---------------|
| Decision Tree       | `max_depth=4`, `criterion='gini'`           | 95.61%        |
| KNN                 | `n_neighbors=9`, `metric='manhattan'`       | 97.37%        |
| Naïve Bayes         | Default                                     | 94.74%        |
| MLP                 | `hidden_layer_sizes=(20,)`, `alpha=0.1`     | 97.37%        |
| SVM                 | `C=10`, `kernel='rbf'`, `gamma=0.01`        | 98.25%        |


### 3. Feature Importance
Top 5 influential features across models:  
1. `concave points_worst` (Decision Tree)  
2. `radius_worst` (SVM)  
3. `perimeter_worst` (KNN)  
4. `concavity_mean` (Naïve Bayes)  
5. `area_se` (MLP)  



## 📦 Saved Models
Pre-trained models available in `models/`:
- `decision_tree_model.pkl`
- `svm_model.pkl` (Best performer)

## 📊 Results Summary
| Model       | Accuracy | Precision | Recall | F1-Score |
|-------------|----------|-----------|--------|----------|
| SVM         | 98.25%   | 98.04%    | 98.25% | 98.14%   |
| MLP         | 97.37%   | 97.22%    | 97.37% | 97.29%   |
| KNN         | 97.37%   | 96.55%    | 97.37% | 96.95%   |

