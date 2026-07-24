# Code for XGBoost & SHAP analysis: Microfluidic Liposome Particle Size Prediction
Python implementation for liposome size prediction and interpretable SHAP analysis in the submitted manuscript.

## Dataset
The raw dataset is obtained from published literature:
Buttitta et al. Machine Learning-Guided microfluidic optimization of liposomes, International Journal of Pharmaceutics, 2025.
Readers can download raw data from the original article.

## Dependencies
Install required packages via:
numpy
pandas
scikit-learn
xgboost
shap
matplotlib

## Functions contained in this code
1. Data processing and feature engineering: construct mechanism-derived features (Total_Lipid, CHOL_ratio, FRR_TFR, PEGylated)
2. Dataset split: 8:2 train / independent test set
3. 5-fold cross validation for Baseline model (original features) and Proposed model (derived features)
4. SHAP value calculation & visualization: feature importance bar plot, SHAP summary plot, SHAP dependence plots

## Notice
All variable names in the source code are consistent with tables and figures in the corresponding manuscript.
