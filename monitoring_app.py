def main():
    st.title("🤖 ML Model Monitoring Dashboard (Test Version)")
    
    # Load sample data
    data = create_sample_data()
    model_info = get_model_info()
    latest = data.iloc[-1]
    
    # Calculate training age
    days_since_training = (datetime.now() - model_info['last_training_date']).days
    days_until_next_training = (model_info['next_training_date'] - datetime.now()).days
    
    # Model info section
    with st.expander("📋 Model Information", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🏷️ Model Details")
            st.write(f"**Model Version:** {model_info['model_version']}")
            st.write(f"**Algorithm:** {model_info['algorithm']}")
            st.write(f"**Training Samples:** {model_info['training_data_size']:,}")
            
        with col2:
            st.markdown("### 📅 Training Information") 
            st.write(f"**Last Training:** {model_info['last_training_date'].strftime('%Y-%m-%d %H:%M')}")
            st.write(f"**Days Ago:** {days_since_training} days")
            st.write(f"**Next Training:** {model_info['next_training_date'].strftime('%Y-%m-%d')}")
            
            if days_until_next_training < 0:
                st.error(f"⚠️ Training overdue by {abs(days_until_next_training)} days!")
            elif days_until_next_training <= 7:
                st.warning(f"🟡 Training due in {days_until_next_training} days")
            else:
                st.success(f"✅ Training scheduled in {days_until_next_training} days")
                
        with col3:
            st.markdown("### 📊 Training Performance")
            st.write(f"**Baseline Accuracy:** {model_info['baseline_accuracy']:.3f}")
            
            baseline_vs_current = latest['accuracy'] - model_info['baseline_accuracy']
            if baseline_vs_current >= 0:
                st.success(f"✅ Current vs Baseline: +{baseline_vs_current:.3f}")
            else:
                st.error(f"📉 Current vs Baseline: {baseline_vs_current:.3f}")
            
            if days_since_training <= 30:
                st.success("🟢 Model is FRESH")
            elif days_since_training <= 60:
                st.warning("🟡 Model is AGING")
            else:
                st.error("🔴 Model is STALE")
    """
Minimal ML Monitoring Dashboard for Free Testing
Copy this code to any of the free platforms above.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="ML Monitoring Test", layout="wide")

@st.cache_data
def create_sample_data():
    """Create sample monitoring data for testing."""
    dates = pd.date_range(start='2025-01-01', end='2025-01-15', freq='D')
    
    data = []
    for i, date in enumerate(dates):
        # Simulate performance degradation over time
        base_accuracy = 0.9 - (i * 0.01) + np.random.normal(0, 0.02)
        drift_score = min(0.3, i * 0.02 + np.random.normal(0, 0.01))
        
        data.append({
            'date': date,
            'accuracy': max(0.5, base_accuracy),
            'drift_score': max(0, drift_score),
            'drift_detected': drift_score > 0.1,
            'revenue': np.random.normal(50000, 5000),
            'conversion_rate': max(0, np.random.normal(0.15, 0.02))
        })
    
    return pd.DataFrame(data)

@st.cache_data 
def get_model_info():
    """Get model training information."""
    return {
        'last_training_date': datetime(2024, 12, 15, 14, 30),
        'model_version': 'v2.1.0',
        'baseline_accuracy': 0.892,
        'training_data_size': 125000,
        'next_training_date': datetime(2025, 2, 15, 9, 0),
        'algorithm': 'Random Forest Classifier'
    }

def main():
    st.title("🤖 ML Model Monitoring Dashboard (Test Version)")
    
    # Load sample data
    data = create_sample_data()
    latest = data.iloc[-1]
    
    # Metrics overview with training age
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Model Accuracy", f"{latest['accuracy']:.3f}", 
                 delta=f"{latest['accuracy'] - data.iloc[-2]['accuracy']:.3f}")
    
    with col2:
        status = "🚨 DETECTED" if latest['drift_detected'] else "✅ NORMAL"
        st.metric("Data Drift", status, f"Score: {latest['drift_score']:.3f}")
    
    with col3:
        st.metric("Daily Revenue", f"${latest['revenue']:,.0f}")
    
    with col4:
        st.metric("Conversion Rate", f"{latest['conversion_rate']:.1%}")
    
    with col5:
        # Training age status
        if days_since_training <= 30:
            age_status = "🟢 FRESH"
            age_color = "green"
        elif days_since_training <= 60:
            age_status = "🟡 AGING" 
            age_color = "orange"
        else:
            age_status = "🔴 STALE"
            age_color = "red"
            
        st.metric("Model Age", age_status, f"{days_since_training} days")
    
    # Charts
    tab1, tab2, tab3 = st.tabs(["Performance", "Drift Analysis", "Business Impact"])
    
    with tab1:
        st.subheader("Model Performance Over Time")
        fig1 = px.line(data, x='date', y='accuracy', title='Accuracy Trend')
        fig1.add_hline(y=0.8, line_dash="dash", line_color="red", 
                      annotation_text="Threshold")
        # Add baseline accuracy line
        fig1.add_hline(y=model_info['baseline_accuracy'], line_dash="dot", 
                      line_color="green", annotation_text=f"Training Baseline: {model_info['baseline_accuracy']:.3f}")
        st.plotly_chart(fig1, use_container_width=True)
        
        # Performance vs baseline comparison
        st.subheader("Performance Comparison")
        col1, col2 = st.columns(2)
        with col1:
            current_accuracy = latest['accuracy']
            baseline_accuracy = model_info['baseline_accuracy']
            accuracy_delta = current_accuracy - baseline_accuracy
            
            st.metric("Current Accuracy", f"{current_accuracy:.3f}")
            st.metric("Baseline Accuracy", f"{baseline_accuracy:.3f}")
            
            if accuracy_delta >= 0:
                st.success(f"Performance vs Baseline: +{accuracy_delta:.3f}")
            else:
                st.error(f"Performance vs Baseline: {accuracy_delta:.3f}")
                
        with col2:
            # Training timeline
            st.write("**Training Timeline:**")
            st.write(f"Last Training: {model_info['last_training_date'].strftime('%Y-%m-%d')}")
            st.write(f"Next Training: {model_info['next_training_date'].strftime('%Y-%m-%d')}")
            st.write(f"Training Frequency: ~{(model_info['next_training_date'] - model_info['last_training_date']).days} days")
    
    with tab2:
        st.subheader("Data Drift Monitoring")
        
        # Drift score over time
        colors = ['red' if x else 'green' for x in data['drift_detected']]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=data['date'], 
            y=data['drift_score'],
            mode='markers+lines',
            marker=dict(color=colors),
            name='Drift Score'
        ))
        fig2.add_hline(y=0.1, line_dash="dash", line_color="orange", 
                      annotation_text="Alert Threshold")
        fig2.update_layout(title="Drift Score Trend")
        st.plotly_chart(fig2, use_container_width=True)
        
        # Drift events
        drift_events = data[data['drift_detected']]
        if not drift_events.empty:
            st.error(f"⚠️ {len(drift_events)} drift events detected!")
            st.dataframe(drift_events[['date', 'drift_score']], use_container_width=True)
            
            # Drift vs training age correlation
            if days_since_training > 30:
                st.warning(f"🕒 **Note:** Model is {days_since_training} days old. Drift may be related to model staleness.")
        else:
            st.success("✅ No drift detected in this period")
    
    with tab3:
        st.subheader("Business Impact")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig3 = px.line(data, x='date', y='revenue', title='Revenue Trend')
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            fig4 = px.line(data, x='date', y='conversion_rate', title='Conversion Rate')
            st.plotly_chart(fig4, use_container_width=True)
        
        # Business summary with training context
        total_revenue = data['revenue'].sum()
        avg_conversion = data['conversion_rate'].mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Period Revenue", f"${total_revenue:,.0f}")
        with col2:
            st.metric("Average Conversion Rate", f"{avg_conversion:.1%}")
        with col3:
            # Show revenue impact of potential model staleness
            if days_since_training > 60:
                potential_loss = total_revenue * 0.05  # Assume 5% loss due to staleness
                st.metric("Potential Revenue Risk", f"${potential_loss:,.0f}", 
                         help="Estimated revenue at risk due to model staleness")
    
    # Enhanced alerts section
    st.header("🚨 Active Alerts")
    alert_count = 0
    
    # Training-related alerts
    if days_until_next_training < 0:
        st.error(f"📅 **Training Overdue** - Scheduled training is {abs(days_until_next_training)} days overdue!")
        alert_count += 1
    elif days_since_training > 90:
        st.error(f"🕒 **Model Age Critical** - Model is {days_since_training} days old and needs retraining")
        alert_count += 1
    elif days_since_training > 60:
        st.warning(f"🕒 **Model Age Warning** - Model is {days_since_training} days old, consider retraining")
        alert_count += 1
    
    # Performance alerts  
    baseline_drop = model_info['baseline_accuracy'] - latest['accuracy']
    if baseline_drop > 0.05:
        st.error(f"📊 **Significant Performance Drop** - Accuracy dropped {baseline_drop:.3f} from training baseline")
        alert_count += 1
    elif baseline_drop > 0.02:
        st.warning(f"📊 **Performance Drop** - Accuracy dropped {baseline_drop:.3f} from training baseline")
        alert_count += 1
    
    # Existing alerts
    if latest['drift_detected']:
        st.error("⚠️ **Data Drift Detected** - Model inputs have shifted significantly from training distribution")
        alert_count += 1
    
    if latest['accuracy'] < 0.75:
        st.error("📉 **Performance Degradation** - Model accuracy has dropped below acceptable threshold")
        alert_count += 1
    
    if latest['conversion_rate'] < 0.10:
        st.warning("💰 **Business Impact Alert** - Conversion rates are below expected levels")
        alert_count += 1
    
    if alert_count == 0:
        st.success("✅ **All Systems Healthy** - No active alerts or issues detected")
    else:
        st.info(f"📋 **Summary:** {alert_count} alert(s) require attention")
    
    # Raw data (optional)
    with st.expander("View Raw Data"):
        # Add model info to the display
        st.write("**Model Information:**")
        st.json(model_info, expanded=False)
        st.write("**Monitoring Data:**")
        st.dataframe(data, use_container_width=True)

if __name__ == "__main__":
    main()
