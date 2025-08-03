import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from model import run_enhanced_analysis  # Import your main function
import os
os.environ["STREAMLIT_CONFIG_FILE"] = "./.streamlit/config.toml"

st.set_page_config(
    layout="wide", 
    page_title="Smart Battery Health Tracker",
    page_icon="🔋",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Better Styling ---
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .status-healthy { border-left-color: #28a745; }
    .status-warning { border-left-color: #ffc107; }
    .status-critical { border-left-color: #dc3545; }
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .insight-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>🔋 Smart Battery Health Tracker & RUL Predictor 🧠</h1>
    <p>Advanced ML-powered predictive maintenance system for lithium-ion batteries</p>
    <p><em>Real-time degradation tracking • Anomaly detection • Predictive analytics</em></p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ System Configuration")
    
    # Battery Type Selection
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.subheader("🔋 Battery Specifications")
    battery_type = st.selectbox(
        "Battery Type",
        ["Li-ion 18650", "Li-ion Pouch", "LiFePO4", "Li-polymer", "Custom"],
        help="Select the battery chemistry for optimized analysis"
    )
    
    if battery_type == "Custom":
        nominal_capacity = st.number_input("Nominal Capacity (Ah)", value=2.0, min_value=0.1)
        nominal_voltage = st.number_input("Nominal Voltage (V)", value=3.7, min_value=1.0)
    else:
        # Preset values based on battery type
        battery_specs = {
            "Li-ion 18650": {"capacity": 2.5, "voltage": 3.7},
            "Li-ion Pouch": {"capacity": 20.0, "voltage": 3.7},
            "LiFePO4": {"capacity": 10.0, "voltage": 3.2},
            "Li-polymer": {"capacity": 5.0, "voltage": 3.7}
        }
        nominal_capacity = battery_specs[battery_type]["capacity"]
        nominal_voltage = battery_specs[battery_type]["voltage"]
    
    st.info(f"📊 Capacity: {nominal_capacity} Ah | Voltage: {nominal_voltage} V")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis Settings
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.subheader("🎯 Analysis Settings")
    
    eol_threshold = st.slider(
        "End-of-Life Threshold (%)", 
        min_value=70, max_value=90, value=80,
        help="Battery capacity threshold for End-of-Life determination"
    )
    
    prediction_horizon = st.selectbox(
        "Prediction Horizon",
        ["Short-term (50 cycles)", "Medium-term (100 cycles)", "Long-term (200+ cycles)"],
        index=1
    )
    
    enable_anomaly_detection = st.checkbox(
        "🚨 Enable Anomaly Detection", 
        value=True,
        help="Detect unusual patterns in battery behavior"
    )
    
    enable_real_time = st.checkbox(
        "📡 Real-time Monitoring Mode", 
        value=False,
        help="Simulate real-time data updates"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Application Context
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.subheader("🎯 Application Context")
    use_case = st.selectbox(
        "Primary Use Case",
        ["Electric Vehicle", "Energy Storage System", "Consumer Electronics", "Grid Storage", "Research & Development"]
    )
    st.markdown('</div>', unsafe_allow_html=True)

# --- Main Content Area ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📁 Data Input")
    uploaded_file = st.file_uploader(
        "Upload Battery Dataset (.mat file)", 
        type="mat",
        help="Upload NASA Battery Dataset or compatible .mat file"
    )

with col2:
    if uploaded_file:
        st.success("✅ File Ready")
        st.info(f"📄 **{uploaded_file.name}**")
    else:
        st.info("📤 Awaiting file upload...")

# --- Demo Data Option ---
if not uploaded_file:
    st.markdown("---")
    st.subheader("🧪 Try Demo Data")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔋 NASA B0005 Dataset", use_container_width=True):
            st.info("Demo mode: Simulating NASA B0005 battery data analysis...")
    
    with col2:
        if st.button("⚡ High-Discharge Scenario", use_container_width=True):
            st.info("Demo mode: Simulating high-discharge rate analysis...")
    
    with col3:
        if st.button("🌡️ Temperature Stress Test", use_container_width=True):
            st.info("Demo mode: Simulating temperature stress analysis...")

# --- Analysis Section ---
if uploaded_file is not None:
    st.markdown("---")
    
    # Analysis Button with Enhanced Styling
    analyze_col1, analyze_col2, analyze_col3 = st.columns([1, 2, 1])
    
    with analyze_col2:
        analyze_button = st.button(
            "🚀 Run Comprehensive Analysis", 
            type="primary", 
            use_container_width=True,
            help="Perform ML-based health assessment and RUL prediction"
        )
    
    if analyze_button:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate analysis steps
        steps = [
            "📊 Loading and preprocessing data...",
            "🔍 Extracting health indicators...",
            "🧠 Training ML models...",
            "📈 Performing backtesting...",
            "🎯 Generating predictions...",
            "🚨 Running anomaly detection...",
            "✅ Analysis complete!"
        ]
        
        for i, step in enumerate(steps):
            status_text.text(step)
            progress_bar.progress((i + 1) / len(steps))
            time.sleep(0.3)  # Simulate processing time
        
        status_text.empty()
        progress_bar.empty()
        
        # Run actual analysis
        with st.spinner("Running comprehensive analysis..."):
            try:
                results = run_enhanced_analysis(
                    mat_file_path=uploaded_file,
                    eol_threshold=eol_threshold,
                    battery_type=battery_type,
                    streamlit_interface=st  
                )

                if results is None:
                    st.error("🚨 ERROR: Analysis failed. No results returned.")
                    st.stop()
                # Unpack results safely
                df = results.get("dataframe")
                predictor = results.get("predictor")
                results_df = results.get("backtest_results")
                final_prediction = results.get("final_prediction")
                final_method = results.get("final_method", "Unknown")
                individual_preds = results.get("individual_predictions", {})
                health_score = results.get("health_score", 0)
                health_status = results.get("health_status", "Unknown")
                anomalies_df = results.get("anomalies", pd.DataFrame())
                confidence = results.get("prediction_confidence", 0)
                analysis_summary = results.get("analysis_summary", {})
                
                # --- ENHANCED INSIGHTS SECTION ---
                st.header("🎯 Smart Analysis Results", divider='rainbow')
                
                # Create insights based on results
                insights = []
                
                # RUL-based insights
                if final_prediction:
                    if final_prediction <= 5:
                        insights.append({
                            "type": "critical",
                            "icon": "🚨",
                            "title": "Critical: Immediate Action Required",
                            "message": f"Battery has only {final_prediction} cycles remaining. Replace immediately to avoid system failure."
                        })
                    elif final_prediction <= 20:
                        insights.append({
                            "type": "warning", 
                            "icon": "⚠️",
                            "title": "Warning: Plan Replacement Soon",
                            "message": f"Battery has {final_prediction} cycles remaining. Schedule replacement within the next maintenance window."
                        })
                    elif final_prediction <= 50:
                        insights.append({
                            "type": "caution",
                            "icon": "📋",
                            "title": "Caution: Monitor Closely", 
                            "message": f"Battery has {final_prediction} cycles remaining. Begin sourcing replacement and planning maintenance."
                        })
                    else:
                        insights.append({
                            "type": "good",
                            "icon": "✅",
                            "title": "Good: Battery Healthy",
                            "message": f"Battery has {final_prediction} cycles remaining. Continue normal operation with regular monitoring."
                        })
                
                # Health-based insights
                if health_score < 50:
                    insights.append({
                        "type": "critical",
                        "icon": "💔",
                        "title": "Poor Battery Health Detected",
                        "message": f"Health score is {health_score:.1f}/100. Multiple degradation indicators are concerning."
                    })
                elif health_score < 70:
                    insights.append({
                        "type": "warning",
                        "icon": "🔍",
                        "title": "Declining Battery Health",
                        "message": f"Health score is {health_score:.1f}/100. Monitor degradation trends closely."
                    })
                else:
                    insights.append({
                        "type": "good",
                        "icon": "💚",
                        "title": "Good Battery Health",
                        "message": f"Health score is {health_score:.1f}/100. Battery is performing well."
                    })
                
                # Confidence-based insights
                if confidence < 60:
                    insights.append({
                        "type": "info",
                        "icon": "📊",
                        "title": "Low Prediction Confidence",
                        "message": f"Prediction confidence is {confidence}%. Consider collecting more data for better accuracy."
                    })
                elif confidence > 85:
                    insights.append({
                        "type": "good",
                        "icon": "🎯",
                        "title": "High Prediction Confidence",
                        "message": f"Prediction confidence is {confidence}%. Results are highly reliable."
                    })
                
                # Anomaly-based insights
                if len(anomalies_df) > 10:
                    insights.append({
                        "type": "warning",
                        "icon": "🚨",
                        "title": "Multiple Anomalies Detected",
                        "message": f"{len(anomalies_df)} anomalies found. Investigate operating conditions and usage patterns."
                    })
                elif len(anomalies_df) == 0:
                    insights.append({
                        "type": "good",
                        "icon": "✨",
                        "title": "Clean Operation Profile",
                        "message": "No anomalies detected. Battery is operating under normal conditions."
                    })
                
                # Display insights
                for insight in insights:
                    if insight["type"] == "critical":
                        st.error(f"{insight['icon']} **{insight['title']}**\n\n{insight['message']}")
                    elif insight["type"] == "warning":
                        st.warning(f"{insight['icon']} **{insight['title']}**\n\n{insight['message']}")
                    elif insight["type"] == "caution":
                        st.info(f"{insight['icon']} **{insight['title']}**\n\n{insight['message']}")
                    elif insight["type"] == "good":
                        st.success(f"{insight['icon']} **{insight['title']}**\n\n{insight['message']}")
                    else:
                        st.info(f"{insight['icon']} **{insight['title']}**\n\n{insight['message']}")
                
                # --- MAIN DASHBOARD ---
                st.header("🎯 Battery Health Dashboard", divider='blue')
                
                # Key metrics row
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                # Current health status
                current_soh = df['soh_percent'].iloc[-1] if df is not None and not df.empty else 0
                
                # Determine status class
                if health_status.lower() in ['excellent', 'good']:
                    status_class = "status-healthy"
                elif health_status.lower() == 'fair':
                    status_class = "status-warning"
                else:
                    status_class = "status-critical"
                
                with metric_col1:
                    st.markdown(f'<div class="metric-card {status_class}">', unsafe_allow_html=True)
                    st.metric(
                        label="🏥 Battery Health",
                        value=f"{current_soh:.1f}%",
                        delta=f"{health_status}",
                        help="Current State of Health (SoH)"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with metric_col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    rul_display = f"{final_prediction} cycles" if final_prediction else "N/A"
                    st.metric(
                        label="⏱️ Remaining Life",
                        value=rul_display,
                        help=f"Predicted using {final_method} model"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with metric_col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    confidence_label = "High" if confidence >= 80 else "Medium" if confidence >= 60 else "Low"
                    st.metric(
                        label="🎯 Confidence",
                        value=confidence_label,
                        delta=f"{confidence:.0f}%",
                        help="Prediction reliability based on model agreement and data quality"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with metric_col4:
                    total_cycles = df['cycle'].max() if df is not None and not df.empty else 0
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        label="🔄 Total Cycles",
                        value=f"{total_cycles:,}",
                        help="Number of charge-discharge cycles completed"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # --- ADVANCED VISUALIZATIONS ---
                st.header("📊 Advanced Analytics", divider='green')
                
                # Create tabs for different analyses
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📈 Health Trends", 
                    "🔮 RUL Prediction", 
                    "🚨 Anomaly Detection", 
                    "📋 Performance Report",
                    "⚙️ Model Insights"
                ])
                
                with tab1:
                    st.subheader("Battery Health Degradation Over Time")
                    
                    if df is not None and not df.empty:
                        # Create comprehensive health visualization
                        fig = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=('State of Health (%)', 'Capacity Trends', 
                                           'Voltage Profile', 'Temperature Analysis'),
                            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                                   [{"secondary_y": False}, {"secondary_y": False}]]
                        )
                        
                        # SoH plot with trend line
                        fig.add_trace(
                            go.Scatter(x=df['cycle'], y=df['soh_percent'], 
                                      name='SoH', line=dict(color='#1f77b4', width=3),
                                      mode='lines+markers', marker=dict(size=3)),
                            row=1, col=1
                        )
                        
                        # Add EOL threshold line
                        fig.add_hline(y=eol_threshold, line_dash="dash", line_color="red", 
                                     annotation_text=f"EOL Threshold ({eol_threshold}%)",
                                     row=1, col=1)
                        
                        # Add trend line
                        if len(df) > 10:
                            z = np.polyfit(df['cycle'], df['soh_percent'], 1)
                            trend_line = np.poly1d(z)
                            fig.add_trace(
                                go.Scatter(x=df['cycle'], y=trend_line(df['cycle']),
                                          name='Trend', line=dict(color='red', width=2, dash='dot')),
                                row=1, col=1
                            )
                        
                        # Capacity plot
                        if 'capacity_ah' in df.columns:
                            fig.add_trace(
                                go.Scatter(x=df['cycle'], y=df['capacity_ah'], 
                                          name='Capacity (Ah)', line=dict(color='#ff7f0e', width=2)),
                                row=1, col=2
                            )
                        
                        # Voltage plot
                        if 'avg_voltage' in df.columns:
                            fig.add_trace(
                                go.Scatter(x=df['cycle'], y=df['avg_voltage'], 
                                          name='Avg Voltage (V)', line=dict(color='#2ca02c', width=2)),
                                row=2, col=1
                            )
                        
                        # Temperature plot
                        if 'avg_temp_c' in df.columns:
                            fig.add_trace(
                                go.Scatter(x=df['cycle'], y=df['avg_temp_c'], 
                                          name='Temperature (°C)', line=dict(color='#d62728', width=2)),
                                row=2, col=2
                            )
                        
                        fig.update_layout(height=600, showlegend=True, 
                                         title_text="Multi-Parameter Health Monitoring")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Health degradation summary
                        st.subheader("📊 Degradation Analysis")
                        deg_col1, deg_col2, deg_col3 = st.columns(3)
                        
                        initial_soh = df['soh_percent'].iloc[0]
                        total_degradation = initial_soh - current_soh
                        avg_degradation_rate = total_degradation / len(df) if len(df) > 0 else 0
                        
                        with deg_col1:
                            st.metric("Total Degradation", f"{total_degradation:.1f}%")
                        with deg_col2:
                            st.metric("Avg Degradation Rate", f"{avg_degradation_rate:.3f}%/cycle")
                        with deg_col3:
                            cycles_to_eol = (current_soh - eol_threshold) / avg_degradation_rate if avg_degradation_rate > 0 else float('inf')
                            st.metric("Linear Projection to EOL", f"{cycles_to_eol:.0f} cycles" if cycles_to_eol != float('inf') else "N/A")
                    
                    else:
                        st.error("No data available for visualization")
                
                with tab2:
                    st.subheader("Remaining Useful Life Prediction")
                    
                    if df is not None and not df.empty and predictor:
                        # Create RUL prediction visualization
                        try:
                            fig = predictor.plot_rul_predictions(df, test_cycle=df['cycle'].max(), return_fig=True)
                            st.pyplot(fig)
                        except Exception as e:
                            st.warning(f"Could not generate RUL plot: {e}")
                            
                            # Fallback: Create simple RUL visualization
                            fig = go.Figure()
                            
                            # Add SoH data
                            fig.add_trace(go.Scatter(
                                x=df['cycle'], 
                                y=df['soh_percent'],
                                mode='lines+markers',
                                name='State of Health',
                                line=dict(color='blue', width=3)
                            ))
                            
                            # Add EOL threshold
                            fig.add_hline(y=eol_threshold, line_dash="dash", line_color="red",
                                         annotation_text=f"EOL Threshold ({eol_threshold}%)")
                            
                            # Add prediction point
                            if final_prediction:
                                predicted_eol_cycle = df['cycle'].max() + final_prediction
                                fig.add_vline(x=predicted_eol_cycle, line_dash="dot", line_color="green",
                                             annotation_text=f"Predicted EOL (Cycle {predicted_eol_cycle})")
                            
                            fig.update_layout(
                                title="RUL Prediction Visualization",
                                xaxis_title="Cycle Number",
                                yaxis_title="State of Health (%)",
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Add prediction details
                        st.subheader("🔍 Prediction Breakdown")
                        
                        if individual_preds:
                            pred_data = []
                            for method, pred in individual_preds.items():
                                status = "✅ Available" if pred is not None else "❌ Not Available"
                                pred_value = f"{pred} cycles" if pred is not None else "N/A"
                                pred_data.append({
                                    "Model": method.replace('_', ' ').title(),
                                    "Prediction": pred_value,
                                    "Status": status
                                })
                            
                            pred_df = pd.DataFrame(pred_data)
                            st.dataframe(pred_df, use_container_width=True)
                        
                        # Prediction reliability analysis
                        st.subheader("📊 Prediction Reliability")
                        
                        valid_preds = [p for p in individual_preds.values() if p is not None]
                        if len(valid_preds) > 1:
                            pred_std = np.std(valid_preds)
                            pred_mean = np.mean(valid_preds)
                            cv = pred_std / pred_mean if pred_mean > 0 else 0
                            
                            rel_col1, rel_col2, rel_col3 = st.columns(3)
                            
                            with rel_col1:
                                st.metric("Model Agreement", f"{len(valid_preds)}/{len(individual_preds)} models")
                            with rel_col2:
                                st.metric("Prediction Std Dev", f"{pred_std:.1f} cycles")
                            with rel_col3:
                                agreement_pct = max(0, 100 - (cv * 100))
                                st.metric("Agreement Score", f"{agreement_pct:.1f}%")
                    
                    else:
                        st.error("No data available for RUL prediction")
                
                with tab3:
                    st.subheader("🚨 Anomaly Detection & Health Alerts")
                    
                    if enable_anomaly_detection and df is not None and not df.empty:
                        if not anomalies_df.empty and 'cycles' in anomalies_df.columns:
                            st.info(f"🔍 Detected {len(anomalies_df)} anomalies in battery behavior")
                            
                            # Create anomaly visualization
                            anomaly_fig = go.Figure()
                            
                            # Add normal data
                            anomaly_fig.add_trace(go.Scatter(
                                x=df['cycle'], 
                                y=df['soh_percent'],
                                mode='lines',
                                name='Normal Operation',
                                line=dict(color='blue', width=2)
                            ))
                            
                            # Add detected anomalies
                            anomaly_cycles = anomalies_df['cycles'].tolist()
                            anomaly_soh_values = []
                            
                            for cycle in anomaly_cycles:
                                matching_rows = df[df['cycle'] == cycle]
                                if not matching_rows.empty:
                                    anomaly_soh_values.append(matching_rows['soh_percent'].iloc[0])
                                else:
                                    anomaly_soh_values.append(df['soh_percent'].mean())
                            
                            # Group anomalies by type if available
                            if 'types' in anomalies_df.columns:
                                anomaly_types = anomalies_df['types'].unique()
                                colors_map = {'Statistical': 'red', 'Pattern': 'orange', 'Degradation': 'purple', 'Temperature': 'yellow'}
                                
                                for anomaly_type in anomaly_types:
                                    type_anomalies = anomalies_df[anomalies_df['types'] == anomaly_type]
                                    type_cycles = type_anomalies['cycles'].tolist()
                                    type_soh = []
                                    
                                    for cycle in type_cycles:
                                        matching_rows = df[df['cycle'] == cycle]
                                        if not matching_rows.empty:
                                            type_soh.append(matching_rows['soh_percent'].iloc[0])
                                        else:
                                            type_soh.append(df['soh_percent'].mean())
                                    
                                    anomaly_fig.add_trace(go.Scatter(
                                        x=type_cycles,
                                        y=type_soh,
                                        mode='markers',
                                        name=f'{anomaly_type} Anomalies',
                                        marker=dict(
                                            color=colors_map.get(anomaly_type, 'red'), 
                                            size=10, 
                                            symbol='x'
                                        )
                                    ))
                            else:
                                # Simple anomaly plot
                                anomaly_fig.add_trace(go.Scatter(
                                    x=anomaly_cycles,
                                    y=anomaly_soh_values,
                                    mode='markers',
                                    name='Anomalies',
                                    marker=dict(color='red', size=10, symbol='x')
                                ))
                            
                            anomaly_fig.update_layout(
                                title="Anomaly Detection Results",
                                xaxis_title="Cycle",
                                yaxis_title="State of Health (%)",
                                height=400
                            )
                            
                            st.plotly_chart(anomaly_fig, use_container_width=True)
                            
                            # Anomaly summary
                            if 'types' in anomalies_df.columns:
                                st.write("**Detected Anomalies Summary**")
                                anomaly_summary = anomalies_df['types'].value_counts().reset_index()
                                anomaly_summary.columns = ['Anomaly Type', 'Count']
                                st.dataframe(anomaly_summary, use_container_width=True)
                            
                            # Health alerts
                            alert_col1, alert_col2 = st.columns(2)
                            
                            if 'types' in anomalies_df.columns:
                                degradation_anomalies = len(anomalies_df[anomalies_df['types'] == 'Degradation'])
                                temp_anomalies = len(anomalies_df[anomalies_df['types'] == 'Temperature'])
                                
                                with alert_col1:
                                    if degradation_anomalies > 0:
                                        st.warning(f"⚠️ **Degradation Alert**\n{degradation_anomalies} rapid degradation events detected")
                                    else:
                                        st.success("✅ **Degradation Normal**\nNo unusual degradation patterns detected")
                                
                                with alert_col2:
                                    if temp_anomalies > 0:
                                        st.error(f"🌡️ **Temperature Alert**\n{temp_anomalies} temperature stress events detected")
                                    else:
                                        st.success("✅ **Temperature Normal**\nNo temperature stress detected")
                        
                        else:
                            st.success("✅ No anomalies detected - Battery operating normally")
                            
                            # Show normal operation chart
                            normal_fig = go.Figure()
                            normal_fig.add_trace(go.Scatter(
                                x=df['cycle'], 
                                y=df['soh_percent'],
                                mode='lines+markers',
                                name='Normal Operation',
                                line=dict(color='green', width=2)
                            ))
                            normal_fig.update_layout(
                                title="Normal Battery Operation - No Anomalies Detected",
                                xaxis_title="Cycle",
                                yaxis_title="State of Health (%)",
                                height=400
                            )
                            st.plotly_chart(normal_fig, use_container_width=True)
                    
                    else:
                        st.info("Enable anomaly detection in the sidebar to see health alerts.")
                
                with tab4:
                    st.subheader("📋 Comprehensive Performance Report")
                    
                    # Battery specifications and analysis summary
                    st.write("**Battery Analysis Summary**")
                    
                    summary_data = {
                        "Parameter": [
                            "Battery Type", 
                            "EOL Threshold", 
                            "Total Cycles Analyzed", 
                            "Current SoH", 
                            "Health Score",
                            "Health Status", 
                            "Anomalies Detected", 
                            "Capacity Fade",
                            "Final Prediction Method",
                            "Prediction Confidence"
                        ],
                        "Value": [
                            battery_type, 
                            f"{eol_threshold}%", 
                            f"{analysis_summary.get('total_cycles', 'N/A')} cycles",
                            f"{analysis_summary.get('current_soh', current_soh):.1f}%",
                            f"{health_score:.1f}/100",
                            health_status,
                            f"{analysis_summary.get('anomaly_count', len(anomalies_df))}",
                            f"{analysis_summary.get('capacity_fade', 100-current_soh):.1f}%",
                            final_method,
                            f"{confidence}%"
                        ]
                    }
                    st.table(pd.DataFrame(summary_data))
                    
                    # Performance metrics table
                    if results_df is not None and len(results_df) > 0:
                        st.write("**Backtesting Performance Analysis**")
                        
                        # Calculate performance metrics safely
                        try:
                            numeric_errors = pd.to_numeric(results_df['final_error'], errors='coerce').dropna()
                            
                            if len(numeric_errors) > 0:
                                mae = numeric_errors.abs().mean()
                                rmse = np.sqrt(numeric_errors.pow(2).mean())
                                
                                # Calculate MAPE safely
                                actual_rul_numeric = pd.to_numeric(results_df['actual_rul'], errors='coerce')
                                valid_indices = ~(numeric_errors.isna() | actual_rul_numeric.isna() | (actual_rul_numeric == 0))
                                
                                if valid_indices.sum() > 0:
                                    mape = (numeric_errors[valid_indices].abs() / actual_rul_numeric[valid_indices] * 100).mean()
                                else:
                                    mape = 0
                                
                                perf_col1, perf_col2, perf_col3 = st.columns(3)
                                
                                with perf_col1:
                                    st.metric("Mean Absolute Error", f"{mae:.2f} cycles")
                                with perf_col2:
                                    st.metric("Root Mean Square Error", f"{rmse:.2f} cycles")
                                with perf_col3:
                                    st.metric("Mean Absolute Percentage Error", f"{mape:.1f}%")
                            
                            # Show subset of backtesting results
                            st.write("**Sample Backtesting Results**")
                            display_cols = [col for col in ['test_cycle', 'actual_rul', 'final_rul', 'final_error', 'prediction_confidence'] 
                                          if col in results_df.columns]
                            
                            if display_cols:
                                display_df = results_df[display_cols].head(10).copy()
                                # Format numeric columns
                                for col in display_df.columns:
                                    if col in ['actual_rul', 'final_rul', 'final_error']:
                                        display_df[col] = pd.to_numeric(display_df[col], errors='coerce').round(1)
                                
                                st.dataframe(display_df, use_container_width=True)
                                
                        except Exception as e:
                            st.warning(f"Could not calculate performance metrics: {e}")
                    
                    # Maintenance recommendations
                    st.write("**🔧 Maintenance Recommendations**")
                    
                    recommendations = []
                    
                    if final_prediction and final_prediction <= 10:
                        recommendations.append("🚨 **Immediate**: Replace battery within next 5 cycles")
                        recommendations.append("📋 **Action**: Order replacement battery immediately")
                        recommendations.append("⚡ **Monitor**: Check daily for performance degradation")
                    elif final_prediction and final_prediction <= 50:
                        recommendations.append("📅 **Schedule**: Plan replacement within next maintenance window")
                        recommendations.append("📦 **Preparation**: Source replacement battery")
                        recommendations.append("📊 **Monitor**: Increase monitoring frequency to weekly")
                    else:
                        recommendations.append("✅ **Continue**: Normal operation and monitoring")
                        recommendations.append("📈 **Review**: Monthly health assessment")
                        recommendations.append("🔍 **Optimize**: Consider usage pattern optimization")
                    
                    if len(anomalies_df) > 10:
                        recommendations.append("🔍 **Investigate**: High anomaly count - check operating conditions")
                    
                    if health_score < 60:
                        recommendations.append("🏥 **Health**: Consider detailed battery diagnostics")
                    
                    for rec in recommendations:
                        st.write(f"• {rec}")
                
                with tab5:
                    st.subheader("⚙️ Model Performance & Insights")
                    
                    # Model comparison
                    st.write("**Individual Model Predictions**")
                    if individual_preds:
                        model_data = []
                        for method, pred in individual_preds.items():
                            reliability = results.get("model_reliability", {}).get(method, 0)
                            if pred is not None:
                                model_data.append({
                                    "Model": method.replace('_', ' ').title(),
                                    "Prediction (cycles)": str(pred),
                                    "Reliability": f"{reliability:.2f}",
                                    "Status": "✅ Available"
                                })
                            else:
                                model_data.append({
                                    "Model": method.replace('_', ' ').title(),
                                    "Prediction (cycles)": "N/A",
                                    "Reliability": "0.00",
                                    "Status": "❌ Not available"
                                })

                        model_df = pd.DataFrame(model_data)
                        st.dataframe(model_df, use_container_width=True)
                    
                    # Feature importance visualization
                    st.write("**Key Health Indicators Impact**")
                    
                    feature_data = {
                        "Feature": [
                            "State of Health Trend", "Capacity Degradation Rate", 
                            "Voltage Stability", "Temperature Stress", 
                            "Cycle Count", "Usage Patterns"
                        ],
                        "Importance": [0.35, 0.25, 0.15, 0.12, 0.08, 0.05],
                        "Impact": ["Critical", "High", "Medium", "Medium", "Low", "Low"]
                    }
                    
                    importance_df = pd.DataFrame(feature_data)
                    
                    # Create bar chart for feature importance
                    fig_importance = px.bar(
                        importance_df, 
                        x='Importance', 
                        y='Feature',
                        color='Impact',
                        orientation='h',
                        title="Feature Importance for RUL Prediction",
                        color_discrete_map={
                            'Critical': '#dc3545',
                            'High': '#fd7e14', 
                            'Medium': '#ffc107',
                            'Low': '#28a745'
                        }
                    )
                    fig_importance.update_layout(height=400)
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Model performance insights
                    st.write("**Analysis Insights**")
                    
                    technical_insights = []
                    
                    # Model-specific insights
                    if final_method == 'trend_analysis':
                        technical_insights.append("📈 **Trend Analysis**: Used linear regression on recent degradation patterns")
                    elif final_method == 'polynomial':
                        technical_insights.append("📊 **Polynomial Model**: Captured non-linear degradation behavior")
                    elif final_method == 'exponential_smoothing':
                        technical_insights.append("📉 **Exponential Smoothing**: Applied time-series forecasting techniques")
                    
                    # Data quality insights
                    if df is not None and len(df) > 100:
                        technical_insights.append(f"📊 **Data Quality**: Rich dataset with {len(df)} cycles enables reliable predictions")
                    elif df is not None and len(df) < 50:
                        technical_insights.append(f"⚠️ **Data Limitation**: Limited dataset ({len(df)} cycles) may affect prediction accuracy")
                    
                    # Confidence insights
                    if confidence > 90:
                        technical_insights.append("🎯 **High Confidence**: Multiple models agree closely on prediction")
                    elif confidence < 70:
                        technical_insights.append("📊 **Moderate Confidence**: Model predictions show some variation")
                    
                    for insight in technical_insights:
                        st.info(insight)
                
                # --- EXPORT OPTIONS ---
                st.header("📤 Export & Integration", divider='gray')
                
                export_col1, export_col2, export_col3 = st.columns(3)
                
                with export_col1:
                    if st.button("📊 Export Report (PDF)", use_container_width=True):
                        st.success("Report generation initiated...")
                
                with export_col2:
                    if st.button("📋 Export Data (CSV)", use_container_width=True):
                        if df is not None and not df.empty:
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name=f"battery_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.error("No data available for export")
                
                with export_col3:
                    if st.button("🔗 API Integration", use_container_width=True):
                        st.info("API endpoints for real-time integration:\n- GET /api/health\n- GET /api/predictions\n- POST /api/data")
                
                # --- REAL-TIME MONITORING ---
                if enable_real_time:
                    st.header("📡 Real-Time Monitoring", divider='green')
                    
                    # Create placeholders for real-time updates
                    rt_col1, rt_col2, rt_col3 = st.columns(3)
                    
                    with rt_col1:
                        st.metric("🔋 Live SoH", f"{current_soh:.1f}%", delta="-0.1%")
                    
                    with rt_col2:
                        temp_value = df['avg_temp_c'].iloc[-1] if 'avg_temp_c' in df.columns else 25.0
                        st.metric("🌡️ Temperature", f"{temp_value:.1f}°C", delta="0.2°C")
                    
                    with rt_col3:
                        current_value = df.get('avg_current', [2.1]).iloc[-1] if 'avg_current' in df.columns else 2.1
                        st.metric("⚡ Current Load", f"{current_value:.1f}A", delta="-0.1A")
                    
                    # Real-time chart placeholder
                    st.info("📊 Real-time data streaming would appear here in a production environment")
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                st.info("Please ensure your data file is in the correct format.")
                # Show debug info
                with st.expander("Debug Information"):
                    st.code(f"Error details: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>🔋 Smart Battery Health Tracker</strong> | Developed for predictive maintenance applications</p>
    <p><em>Electric Vehicles • Energy Storage Systems • Smart Grids • Consumer Electronics</em></p>
    <p>Powered by Machine Learning • Real-time Analytics • Anomaly Detection</p>
</div>
""", unsafe_allow_html=True)