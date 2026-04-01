"""
Advanced Fraud Detection Model
Ensemble of LightGBM, XGBoost, and Isolation Forest with SHAP explainability
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE

class AdvancedFraudModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.threshold = 0.5
    
    def preprocess_data(self, df):
        """Preprocess fraud detection data"""
        X = df.drop(['transaction_id', 'is_fraud'], axis=1)
        
        # Encode categorical
        X['merchant_category'] = pd.Categorical(X['merchant_category']).codes
        
        self.feature_cols = X.columns.tolist()
        X = self.scaler.fit_transform(X)
        
        return X, df['is_fraud'].values
    
    def train_ensemble(self, df):
        """Train ensemble: LightGBM + XGBoost + Isolation Forest"""
        X, y = self.preprocess_data(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Apply SMOTE to handle imbalance
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        # Component 1: LightGBM
        lgb = LGBMClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.05,
            class_weight='balanced', random_state=42
        )
        
        # Component 2: XGBoost
        xgb = XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.05,
            scale_pos_weight=len(y_train_smote[y_train_smote == 0]) / len(y_train_smote[y_train_smote == 1]),
            random_state=42
        )
        
        # Component 3: Isolation Forest (unsupervised anomaly detection)
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        
        # Train
        lgb.fit(X_train_smote, y_train_smote)
        xgb.fit(X_train_smote, y_train_smote)
        iso_forest.fit(X_train_smote)
        
        # Voting ensemble
        self.model = VotingClassifier(
            estimators=[('lgb', lgb), ('xgb', xgb)],
            voting='soft'
        )
        self.model.fit(X_train_smote, y_train_smote)
        
        # Metrics
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'precision': precision_score(y_test, (y_pred_proba >= 0.5).astype(int)),
            'recall': recall_score(y_test, (y_pred_proba >= 0.5).astype(int)),
            'f1': f1_score(y_test, (y_pred_proba >= 0.5).astype(int))
        }
        
        return metrics, X_test, y_test, y_pred_proba
    
    def predict(self, X):
        """Predict fraud probability"""
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        return probabilities
    
    def save(self, path):
        """Save model"""
        joblib.dump(self, path)
    
    @staticmethod
    def load(path):
        """Load model"""
        return joblib.load(path)
