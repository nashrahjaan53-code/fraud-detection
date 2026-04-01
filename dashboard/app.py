"""
Fraud Detection Dashboard with SHAP explainability
"""
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generator import FraudDataGenerator, FraudAnalyzer
from src.model import AdvancedFraudModel

st.set_page_config(page_title="Fraud Detection", page_icon="🚨", layout="wide")

@st.cache_resource
def load_model_and_data():
    df = pd.read_csv('data/fraud_data.csv')
    model_path = Path('models/fraud_model.pkl')
    if model_path.exists():
        model = AdvancedFraudModel.load(model_path)
    else:
        model = AdvancedFraudModel()
        metrics, _, _, _ = model.train_ensemble(df)
        model.save(model_path)
    return df, model

df, model = load_model_and_data()
analyzer = FraudAnalyzer(df)

st.title("🚨 Advanced Fraud Detection System")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Predictions", "📈 Analysis", "⚠️ Alerts"])

with tab1:
    st.header("Fraud Detection Dashboard")
    
    stats = analyzer.get_fraud_stats()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", f"{stats['total_transactions']:,}")
    with col2:
        st.metric("Fraud Count", f"{stats['fraud_count']}", delta=f"{stats['fraud_rate']:.3f}%")
    with col3:
        st.metric("Avg. Fraud Amount", f"${stats['avg_fraud_amount']:.2f}")
    with col4:
        st.metric("Avg. Legit Amount", f"${stats['avg_legit_amount']:.2f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Fraud by category
        fraud_by_cat = analyzer.get_fraud_by_category()
        fig = px.bar(fraud_by_cat.reset_index(), x='merchant_category', y='fraud_rate',
                    title='Fraud Rate by Merchant Category',
                    color='fraud_rate', color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Fraud by hour
        fraud_by_hour = analyzer.get_time_patterns()
        fig = px.bar(fraud_by_hour.reset_index(), x='hour', y='fraud_rate',
                    title='Fraud Rate by Hour of Day',
                    color='fraud_rate', color_continuous_scale='Oranges')
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Real-time Risk Scoring")
    
    # Make predictions
    cols_to_drop = [col for col in ['transaction_id', 'is_fraud'] if col in df.columns]
    X = df.drop(cols_to_drop, axis=1)
    if 'merchant_category' in X.columns:
        X['merchant_category'] = pd.Categorical(X['merchant_category']).codes
    
    fraud_probs = model.predict(X)
    df_pred = df.copy()
    df_pred['fraud_probability'] = fraud_probs
    df_pred['risk_tier'] = pd.cut(fraud_probs, bins=[0, 0.3, 0.6, 0.8, 1.0],
                                  labels=['Low', 'Medium', 'High', 'Critical'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk distribution
        risk_counts = df_pred['risk_tier'].value_counts()
        colors = {'Low': '#2ca02c', 'Medium': '#ffbb78', 'High': '#ff7f0e', 'Critical': '#d62728'}
        fig = px.pie(values=risk_counts.values, names=risk_counts.index, title='Risk Distribution',
                    color_discrete_map=colors)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Probability distribution
        fig = px.histogram(df_pred, x='fraud_probability', nbins=50,
                          title='Fraud Probability Distribution',
                          color_discrete_sequence=['#d62728'])
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🔴 High-Risk Transactions (Top 20)")
    
    high_risk = df_pred.nlargest(20, 'fraud_probability')[
        ['transaction_id', 'amount', 'merchant_category', 'fraud_probability', 'risk_tier']
    ].round({'fraud_probability': 3})
    
    st.dataframe(high_risk, use_container_width=True)

with tab3:
    st.header("Detailed Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Amount vs Fraud
        fraud_legit = pd.DataFrame({
            'Type': ['Legitimate', 'Fraudulent'],
            'Avg Amount': [stats['avg_legit_amount'], stats['avg_fraud_amount']]
        })
        fig = px.bar(fraud_legit, x='Type', y='Avg Amount', title='Average Transaction Amount',
                    color_discrete_sequence=['#1f77b4', '#d62728'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Device risk vs fraud
        df_sample = df.sample(min(1000, len(df)))
        fig = px.scatter(df_sample, x='device_risk_score', y='amount', color='is_fraud',
                        title='Device Risk vs Amount',
                        color_discrete_map={0: '#1f77b4', 1: '#d62728'},
                        labels={'is_fraud': 'Fraud'})
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("⚠️ Fraud Alerts & Actions")
    
    # Critical frauds
    critical = df_pred[df_pred['risk_tier'] == 'Critical'].head(10)
    
    if len(critical) > 0:
        st.warning(f"🚨 {len(critical)} CRITICAL TRANSACTIONS DETECTED!")
        
        for idx, row in critical.iterrows():
            with st.expander(f"TXN {row['transaction_id']} - ${row['amount']:.2f}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Fraud Probability:** {row['fraud_probability']:.1%}")
                    st.write(f"**Category:** {row['merchant_category']}")
                    st.write(f"**Amount:** ${row['amount']:.2f}")
                with col2:
                    st.write(f"**Device Risk:** {row['device_risk_score']:.1f}")
                    st.write(f"**Distance from Home:** {row['distance_from_home']:.1f} km")
                    st.write(f"**24h Transactions:** {row['txn_count_24h']}")
                
                st.error("✅ ACTION: Block transaction & notify customer")
    else:
        st.success("✅ No critical transactions detected")
