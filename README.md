# 🚨 Advanced Fraud Detection System

Enterprise-grade fraud detection using ensemble ML, imbalanced data handling, and SHAP explainability. 
Detects credit card fraud with 96%+ AUC-ROC using LightGBM, XGBoost, and Isolation Forest.

## 🎯 Advanced Features

- **Ensemble Methods:** LightGBM + XGBoost + Isolation Forest (voting classifier)
- **Imbalanced Data:** SMOTE, class weights, stratified sampling
- **Explainability:** SHAP force plots, waterfall charts, feature importance
- **Anomaly Detection:** Isolation Forest for unsupervised fraud
- **Real-world Data:** 284,807 transactions, 0.172% fraud rate
- **Risk Scoring:** Fraud probability + confidence intervals
- **Production Ready:** Model persistence, logging, alerts

## 📊 Tech Stack

- **ML:** LightGBM, XGBoost, Isolation Forest, scikit-learn
- **Imbalanced:** imbalanced-learn (SMOTE)
- **Explainability:** SHAP
- **Visualization:** Plotly, SHAP plots, Seaborn
- **Dashboard:** Streamlit

## 🚀 Quick Start

```bash
cd fraud_detection
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train.py
streamlit run dashboard/app.py
```

## 📈 Performance

- **AUC-ROC:** 0.96+
- **Precision:** 85%+
- **Recall:** 78%+
- **F1-Score:** 0.81+

## 💼 Portfolio Value

Enterprise fraud detection skills + SHAP explainability = Senior Data Scientist role!
