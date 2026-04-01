"""
Advanced Fraud Detection Data Generator
Generates realistic credit card fraud data with imbalanced distribution
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class FraudDataGenerator:
    @staticmethod
    def generate_fraud_data(n_samples=10000):
        """Generate realistic credit card fraud data"""
        np.random.seed(42)
        
        data = []
        fraud_count = 0
        fraud_ratio = 0.002  # 0.2% fraud (realistic)
        
        for i in range(n_samples):
            # Time features
            timestamp = datetime.now() - timedelta(days=np.random.randint(0, 365))
            hour = np.random.randint(0, 24)
            day_of_week = timestamp.weekday()
            
            # Transaction amount (log-normal distribution)
            if np.random.random() < fraud_ratio:
                # Fraudulent transaction (higher amounts, unusual patterns)
                amount = np.random.lognormal(3.5, 2)
                fraud_count += 1
                is_fraud = 1
            else:
                # Legitimate transaction
                amount = np.random.lognormal(2.5, 1.2)
                is_fraud = 0
            
            # Merchant category
            merchant_cat = np.random.choice(
                ['Grocery', 'Gas', 'Restaurant', 'Online', 'Travel', 'Entertainment'],
                p=[0.3, 0.2, 0.2, 0.15, 0.1, 0.05]
            )
            
            # Features
            distance_from_home = np.random.exponential(50)
            if is_fraud:
                distance_from_home *= np.random.uniform(2, 5)  # Frauds often far from home
            
            days_since_last_transaction = np.random.exponential(5)
            transaction_frequency = np.random.exponential(10)
            
            # Card age in years
            card_age = np.random.exponential(3)
            
            # Device risk score
            device_risk = np.random.uniform(0, 100)
            if is_fraud:
                device_risk *= np.random.uniform(1.5, 3)
            device_risk = min(device_risk, 100)
            
            # Velocity features
            txn_count_24h = np.random.poisson(3)
            if is_fraud:
                txn_count_24h = np.random.poisson(8)
            
            txn_count_7d = np.random.poisson(20)
            
            data.append({
                'transaction_id': f'TXN_{i:08d}',
                'amount': round(amount, 2),
                'hour': hour,
                'day_of_week': day_of_week,
                'merchant_category': merchant_cat,
                'distance_from_home': round(distance_from_home, 2),
                'days_since_last_txn': round(days_since_last_transaction, 2),
                'transaction_frequency': round(transaction_frequency, 2),
                'card_age_years': round(card_age, 2),
                'device_risk_score': round(device_risk, 2),
                'txn_count_24h': txn_count_24h,
                'txn_count_7d': txn_count_7d,
                'is_fraud': is_fraud
            })
        
        df = pd.DataFrame(data)
        return df

class FraudAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
    
    def get_fraud_stats(self):
        """Get fraud statistics"""
        total = len(self.df)
        fraud = int(self.df['is_fraud'].sum())
        
        return {
            'total_transactions': total,
            'fraud_count': fraud,
            'fraud_rate': round(fraud / total * 100, 3),
            'total_amount': round(self.df['amount'].sum(), 2),
            'fraud_amount': round(self.df[self.df['is_fraud'] == 1]['amount'].sum(), 2),
            'avg_fraud_amount': round(self.df[self.df['is_fraud'] == 1]['amount'].mean(), 2),
            'avg_legit_amount': round(self.df[self.df['is_fraud'] == 0]['amount'].mean(), 2)
        }
    
    def get_fraud_by_category(self):
        """Fraud rate by merchant category"""
        result = self.df.groupby('merchant_category')['is_fraud'].agg(['sum', 'count', 'mean'])
        result.columns = ['fraud_count', 'total', 'fraud_rate']
        return result
    
    def get_time_patterns(self):
        """Fraud by hour of day"""
        result = self.df.groupby('hour')['is_fraud'].agg(['sum', 'count', 'mean'])
        result.columns = ['fraud_count', 'total', 'fraud_rate']
        return result
