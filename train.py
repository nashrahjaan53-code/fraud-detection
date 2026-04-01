"""
Training script for Fraud Detection
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_generator import FraudDataGenerator, FraudAnalyzer
from src.model import AdvancedFraudModel

def main():
    print("🚨 Advanced Fraud Detection Model Training")
    
    # Generate data
    print("\n📊 Generating fraud data...")
    generator = FraudDataGenerator()
    df = generator.generate_fraud_data(n_samples=10000)
    df.to_csv('data/fraud_data.csv', index=False)
    print(f"✓ Generated {len(df):,} transactions")
    
    # Analyze
    analyzer = FraudAnalyzer(df)
    stats = analyzer.get_fraud_stats()
    
    print(f"\n📊 Fraud Statistics:")
    print(f"✓ Total Transactions: {stats['total_transactions']:,}")
    print(f"✓ Fraudulent: {stats['fraud_count']} ({stats['fraud_rate']:.3f}%)")
    print(f"✓ Total Amount: ${stats['total_amount']:,.2f}")
    print(f"✓ Fraud Amount: ${stats['fraud_amount']:,.2f}")
    
    # Train ensemble
    print(f"\n🤖 Training Ensemble Model (LightGBM + XGBoost)...")
    model = AdvancedFraudModel()
    metrics, X_test, y_test, y_pred = model.train_ensemble(df)
    
    print(f"\n✓ Model Performance:")
    print(f"  - AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"  - Precision: {metrics['precision']:.4f}")
    print(f"  - Recall: {metrics['recall']:.4f}")
    print(f"  - F1-Score: {metrics['f1']:.4f}")
    
    # Save
    model.save('models/fraud_model.pkl')
    print(f"\n✅ Model saved!")
    print("🚀 Start dashboard with: streamlit run dashboard/app.py")

if __name__ == '__main__':
    main()
