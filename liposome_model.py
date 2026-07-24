#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[22]:


# ======================== 数据加载 ========================
df = pd.read_csv("formulations.csv")
required_cols = ['ESM', 'HSPC', 'CHOL', 'PEG', 'TFR', 'FRR', 'AQUEOUS', 'SIZE', 'PDI']
df = df[required_cols].dropna()

# ======================== 原文过滤：仅保留成功形成的脂质体 ========================
df = df[(df['SIZE'] <= 500) & (df['PDI'] <= 0.5)].copy()
# 若有芯片列，只保留 Micromixer
if 'CHIP' in df.columns:
    df = df[df['CHIP'] == 'Micromixer']
print(f"After filtering, samples: {len(df)}")

# ======================== 对数变换目标 ========================
df['LOG_SIZE'] = np.log(df['SIZE'])

# ======================== 创新特征工程 ========================
def feature_engineering(df):
    df = df.copy()
    # 1. 总脂质浓度（原文实验因子却未入模）
    df['Total_Lipid'] = df['ESM'] + df['HSPC'] + df['CHOL'] + df['PEG']
    # 2. PEG 化二元标志（Doxil 型有，Marqibo 型无）
    df['PEGylated'] = (df['PEG'] > 0).astype(int)
    # 3. 胆固醇相对含量（膜刚性指标）
    df['CHOL_ratio'] = df['CHOL'] / (df['Total_Lipid'] + 1e-6)
    # 4. 流速交互项（混合强度）
    df['FRR_TFR'] = df['FRR'] * df['TFR']
    return df

df_proposed = feature_engineering(df)

features_base = ['ESM', 'HSPC', 'CHOL', 'PEG', 'TFR', 'FRR', 'AQUEOUS']
target = 'LOG_SIZE'

X_base = df[features_base]
y_base = df[target]
# Proposed 特征集：原始7个 + 新增4个
features_pro = features_base + ['Total_Lipid', 'PEGylated', 'CHOL_ratio', 'FRR_TFR']
X_pro = df_proposed[features_pro]
y_pro = df_proposed[target]


# In[23]:


# ======================== 第二部分：先切出独立测试集 ========================
X_base_cv, X_base_test, y_base_cv, y_base_test = train_test_split(
    X_base, y_base, test_size=0.2, random_state=42)
X_pro_cv, X_pro_test, y_pro_cv, y_pro_test = train_test_split(
    X_pro, y_pro, test_size=0.2, random_state=42)


# In[24]:


# ======================== 第三部分：统一预处理 ========================
categorical_cols = ['AQUEOUS']
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],
    remainder='passthrough'
)

xgb_params = {
    'n_estimators': 200,
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

base_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(**xgb_params))
])
pro_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(**xgb_params))
])


# In[25]:


def rmse_exp(y_true, y_pred):
    """计算原始尺度的 RMSE（预测值指数还原）"""
    y_true_exp = np.exp(y_true)
    y_pred_exp = np.exp(y_pred)
    return np.sqrt(mean_squared_error(y_true_exp, y_pred_exp))

def mae_exp(y_true, y_pred):
    y_true_exp = np.exp(y_true)
    y_pred_exp = np.exp(y_pred)
    return mean_absolute_error(y_true_exp, y_pred_exp)

# 注意：R² 仍然在 log 空间计算，这是合理的，因为模型拟合的是 log 值
scoring = {
    'R2': make_scorer(r2_score),          # log 空间的 R²
    'RMSE': make_scorer(rmse_exp),        # 原始 nm
    'MAE': make_scorer(mae_exp)           # 原始 nm
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

cv_base = cross_validate(base_pipeline, X_base_cv, y_base_cv, cv=cv, scoring=scoring)
cv_pro = cross_validate(pro_pipeline, X_pro_cv, y_pro_cv, cv=cv, scoring=scoring)

def cv_summary(cv_dict):
    return {
        'R2_mean': cv_dict['test_R2'].mean(),
        'R2_std': cv_dict['test_R2'].std(),
        'RMSE_mean': cv_dict['test_RMSE'].mean(),
        'RMSE_std': cv_dict['test_RMSE'].std(),
        'MAE_mean': cv_dict['test_MAE'].mean(),
        'MAE_std': cv_dict['test_MAE'].std()
    }

base_cv = cv_summary(cv_base)
pro_cv = cv_summary(cv_pro)

cv_table = pd.DataFrame({
    'Model': ['Baseline', 'Proposed'],
    'R² (log)': [f"{base_cv['R2_mean']:.3f}±{base_cv['R2_std']:.3f}",
               f"{pro_cv['R2_mean']:.3f}±{pro_cv['R2_std']:.3f}"],
    'RMSE (nm)': [f"{base_cv['RMSE_mean']:.2f}±{base_cv['RMSE_std']:.2f}",
                  f"{pro_cv['RMSE_mean']:.2f}±{pro_cv['RMSE_std']:.2f}"],
    'MAE (nm)': [f"{base_cv['MAE_mean']:.2f}±{base_cv['MAE_std']:.2f}",
                 f"{pro_cv['MAE_mean']:.2f}±{pro_cv['MAE_std']:.2f}"]
})

print("=========================")
print("5-Fold Cross Validation on 80% training set (Mean ± Std)")
print("Note: R² is computed on log-scale; RMSE/MAE are in original nanometers.")
print("=========================")
print(cv_table.to_string(index=False))
cv_table.to_csv("Table2.csv", index=False)


# In[27]:


# ======================== 划分独立测试集 ========================
X_base_cv, X_base_test, y_base_cv, y_base_test = train_test_split(
    X_base, y_base, test_size=0.2, random_state=42)
X_pro_cv, X_pro_test, y_pro_cv, y_pro_test = train_test_split(
    X_pro, y_pro, test_size=0.2, random_state=42)


# In[34]:


# ======================== 分别创建独立的预处理器 ========================
categorical_cols = ['AQUEOUS']

preprocessor_base = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],
    remainder='passthrough'
)

preprocessor_pro = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],
    remainder='passthrough'
)

xgb_params = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 1.0,
    'random_state': 42
}

base_pipeline = Pipeline([
    ('preprocessor', preprocessor_base),
    ('regressor', xgb.XGBRegressor(**xgb_params))
])
pro_pipeline = Pipeline([
    ('preprocessor', preprocessor_pro),
    ('regressor', xgb.XGBRegressor(**xgb_params))
])


# In[35]:


# ======================== 评估指标（还原为纳米） ========================
def rmse_exp(y_true, y_pred):
    y_true_exp = np.exp(y_true)
    y_pred_exp = np.exp(y_pred)
    return np.sqrt(mean_squared_error(y_true_exp, y_pred_exp))

def mae_exp(y_true, y_pred):
    y_true_exp = np.exp(y_true)
    y_pred_exp = np.exp(y_pred)
    return mean_absolute_error(y_true_exp, y_pred_exp)

scoring = {
    'R2': make_scorer(r2_score),
    'RMSE': make_scorer(rmse_exp),
    'MAE': make_scorer(mae_exp)
}


# In[36]:


# ======================== 五折交叉验证 ========================
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_base = cross_validate(base_pipeline, X_base_cv, y_base_cv, cv=cv, scoring=scoring)
cv_pro = cross_validate(pro_pipeline, X_pro_cv, y_pro_cv, cv=cv, scoring=scoring)

def cv_summary(cv_dict):
    return {
        'R2_mean': cv_dict['test_R2'].mean(),
        'R2_std': cv_dict['test_R2'].std(),
        'RMSE_mean': cv_dict['test_RMSE'].mean(),
        'RMSE_std': cv_dict['test_RMSE'].std(),
        'MAE_mean': cv_dict['test_MAE'].mean(),
        'MAE_std': cv_dict['test_MAE'].std()
    }

base_cv = cv_summary(cv_base)
pro_cv = cv_summary(cv_pro)

cv_table = pd.DataFrame({
    'Model': ['Baseline', 'Proposed'],
    'R² (log)': [f"{base_cv['R2_mean']:.3f}±{base_cv['R2_std']:.3f}",
               f"{pro_cv['R2_mean']:.3f}±{pro_cv['R2_std']:.3f}"],
    'RMSE (nm)': [f"{base_cv['RMSE_mean']:.2f}±{base_cv['RMSE_std']:.2f}",
                  f"{pro_cv['RMSE_mean']:.2f}±{pro_cv['RMSE_std']:.2f}"],
    'MAE (nm)': [f"{base_cv['MAE_mean']:.2f}±{base_cv['MAE_std']:.2f}",
                 f"{pro_cv['MAE_mean']:.2f}±{pro_cv['MAE_std']:.2f}"]
})

print("=========================")
print("5-Fold Cross Validation on 80% training set (Mean ± Std)")
print("R² on log-scale; RMSE & MAE in nanometers.")
print("=========================")
print(cv_table.to_string(index=False))
cv_table.to_csv("Table2.csv", index=False)


# In[37]:


final_base = base_pipeline.fit(X_base_cv, y_base_cv)
final_pro = pro_pipeline.fit(X_pro_cv, y_pro_cv)

y_pred_base = final_base.predict(X_base_test)
y_pred_pro = final_pro.predict(X_pro_test)

print("\n=========================")
print("Final Model on Independent Test Set (20%)")
print("=========================")
print(f"Baseline - R² (log): {r2_score(y_base_test, y_pred_base):.4f}, "
      f"RMSE (nm): {rmse_exp(y_base_test, y_pred_base):.2f}, "
      f"MAE (nm): {mae_exp(y_base_test, y_pred_base):.2f}")
print(f"Proposed - R² (log): {r2_score(y_pro_test, y_pred_pro):.4f}, "
      f"RMSE (nm): {rmse_exp(y_pro_test, y_pred_pro):.2f}, "
      f"MAE (nm): {mae_exp(y_pro_test, y_pred_pro):.2f}")


# In[38]:


# ======================== SHAP 分析 ========================
trained_preprocessor = final_pro.named_steps['preprocessor']
X_pro_test_transformed = trained_preprocessor.transform(X_pro_test)
feature_names = trained_preprocessor.get_feature_names_out()
feature_names = [name.replace('remainder__','').replace('cat__','') for name in feature_names]
X_test_df = pd.DataFrame(X_pro_test_transformed, columns=feature_names)

explainer = shap.TreeExplainer(final_pro.named_steps['regressor'])
shap_values = explainer.shap_values(X_test_df)

os.makedirs("figures", exist_ok=True)

plt.figure()
shap.summary_plot(shap_values, X_test_df, show=False)
plt.savefig("figures/shap_summary_beeswarm.pdf", bbox_inches='tight', dpi=300)
plt.close()

plt.figure()
shap.summary_plot(shap_values, X_test_df, plot_type='bar', show=False)
plt.savefig("figures/shap_importance_bar.pdf", bbox_inches='tight', dpi=300)
plt.close()

for feat in ['Total_Lipid', 'CHOL_ratio', 'PEGylated', 'FRR_TFR']:
    if feat in feature_names:
        plt.figure()
        shap.dependence_plot(feat, shap_values, X_test_df, show=False)
        plt.savefig(f"figures/dependence_{feat}.pdf", bbox_inches='tight', dpi=300)
        plt.close()

if 'PEG' in feature_names and 'CHOL' in feature_names:
    plt.figure()
    shap.dependence_plot('PEG', shap_values, X_test_df, interaction_index='CHOL', show=False)
    plt.savefig("figures/interaction_PEG_CHOL.pdf", bbox_inches='tight', dpi=300)
    plt.close()

print("\nSHAP 分析图已保存至 figures/ 文件夹")

