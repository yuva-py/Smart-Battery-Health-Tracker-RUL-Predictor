import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import json
import random
from model import (
    PhysicsInformedBatteryModel, 
    AdvancedBatteryRULPredictor, 
    create_sample_battery_data_with_physics
)
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    layout="wide", 
    page_title="Fleet Battery Intelligence Platform",
    page_icon="🚗⚡",
    initial_sidebar_state="expanded"
)

# --- Enhanced CSS for Fleet Dashboard ---
st.markdown("""
<style>
    .fleet-header {
        text-align: center;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #667eea 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    .vehicle-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .vehicle-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .status-excellent { border-left-color: #28a745; background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); }
    .status-good { border-left-color: #20c997; background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%); }
    .status-fair { border-left-color: #ffc107; background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); }
    .status-poor { border-left-color: #fd7e14; background: linear-gradient(135deg, #ffe8d1 0%, #ffd59a 100%); }
    .status-critical { border-left-color: #dc3545; background: linear-gradient(135deg, #f8d7da 0%, #f1b0b7 100%); }
    
    .physics-insight {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .realtime-metric {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 2px solid #9c27b0;
    }
    .alert-critical {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 8px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    .fleet-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .diagnostic-panel {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #6c757d;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if 'fleet_data' not in st.session_state:
    st.session_state.fleet_data = {}
if 'selected_vehicle' not in st.session_state:
    st.session_state.selected_vehicle = None
if 'real_time_mode' not in st.session_state:
    st.session_state.real_time_mode = False
if 'fleet_initialized' not in st.session_state:
    st.session_state.fleet_initialized = False

# --- Fleet Data Generator ---
def generate_fleet_data(num_vehicles=15):
    """Generate realistic fleet data with physics-informed models"""
    fleet = {}
    
    vehicle_types = ["Model S", "Model 3", "Model X", "Model Y", "ID.4", "EQS", "Taycan"]
    locations = ["Downtown", "Suburb A", "Suburb B", "Airport", "Mall", "Industrial", "Highway"]
    
    for i in range(num_vehicles):
        vehicle_id = f"VEH-{1001 + i}"
        
        # Generate realistic historical data
        cycles = random.randint(150, 800)
        df = create_sample_battery_data_with_physics(cycles=cycles)
        
        # Add some variation for realism
        noise_factor = random.uniform(0.8, 1.2)
        df['soh_percent'] *= noise_factor
        df['soh_percent'] = np.clip(df['soh_percent'], 60, 100)
        
        # Initialize physics model for this vehicle
        physics_model = PhysicsInformedBatteryModel(
            battery_type=random.choice(["Li-ion 18650", "Li-ion Pouch", "LiFePO4"])
        )
        
        # Initialize RUL predictor
        predictor = AdvancedBatteryRULPredictor(
            eol_threshold=80, 
            battery_type=physics_model.battery_type
        )
        
        # Get current state
        current_soh = df['soh_percent'].iloc[-1]
        current_voltage = df['avg_voltage'].iloc[-1]
        current_temp = df['avg_temp_c'].iloc[-1]
        current_current = df['avg_current'].iloc[-1]
        
        # Real-time state simulation
        realtime_state = predictor.process_live_sensor_data(
            current_voltage, current_current, current_temp
        )
        
        # Physics-enhanced RUL prediction
        rul_results = predictor.predict_rul_with_physics(df)
        
        # Generate comprehensive analysis
        comprehensive_results = predictor.analyze_battery_comprehensive_enhanced(df)
        
        fleet[vehicle_id] = {
            'vehicle_id': vehicle_id,
            'vehicle_type': random.choice(vehicle_types),
            'location': random.choice(locations),
            'historical_data': df,
            'physics_model': physics_model,
            'predictor': predictor,
            'current_soh': current_soh,
            'health_status': comprehensive_results['health_status'],
            'health_score': comprehensive_results['health_score'],
            'rul_prediction': rul_results['final_rul'],
            'rul_confidence': rul_results['confidence'],
            'rul_method': rul_results['method_used'],
            'physics_enhanced': rul_results['physics_enhancement'],
            'anomaly_count': comprehensive_results['anomalies']['count'],
            'realtime_state': realtime_state,
            'last_updated': datetime.now(),
            'comprehensive_analysis': comprehensive_results,
            'physics_soh_analysis': rul_results['physics_soh_analysis']
        }
    
    return fleet

# --- Generate Fleet Alerts ---
def generate_fleet_alerts(fleet_data):
    """Generate actionable business insights from fleet analysis"""
    alerts = []
    
    # Critical vehicles needing immediate attention
    critical_vehicles = [v for v in fleet_data.values() if v['rul_prediction'] and v['rul_prediction'] <= 10]
    if critical_vehicles:
        vehicle_list = ", ".join([v['vehicle_id'] for v in critical_vehicles])
        alerts.append({
            'priority': 'CRITICAL',
            'title': 'Immediate Replacement Required',
            'message': f"{len(critical_vehicles)} vehicle(s) need immediate battery replacement: {vehicle_list}",
            'action': 'Schedule immediate maintenance',
            'business_impact': 'High - Risk of service disruption'
        })
    
    # Vehicles with high anomaly counts
    anomaly_vehicles = [v for v in fleet_data.values() if v['anomaly_count'] > 15]
    if anomaly_vehicles:
        alerts.append({
            'priority': 'WARNING',
            'title': 'Unusual Operating Patterns Detected',
            'message': f"{len(anomaly_vehicles)} vehicle(s) showing unusual patterns. May indicate driver behavior or environmental issues.",
            'action': 'Investigate operating conditions',
            'business_impact': 'Medium - Potential accelerated degradation'
        })
    
    # Budget planning alert
    replacement_needed = [v for v in fleet_data.values() if v['rul_prediction'] and v['rul_prediction'] <= 50]
    if replacement_needed:
        alerts.append({
            'priority': 'INFO',
            'title': 'Quarterly Budget Planning',
            'message': f"{len(replacement_needed)} vehicle(s) will need battery replacement within 50 cycles.",
            'action': 'Plan budget for battery replacements',
            'business_impact': 'Planning - Budget allocation needed'
        })
    
    # Physics-enhanced insights
    physics_enhanced = [v for v in fleet_data.values() if v['physics_enhanced']]
    if physics_enhanced:
        alerts.append({
            'priority': 'INFO',
            'title': 'Advanced Physics Analysis Available',
            'message': f"{len(physics_enhanced)} vehicle(s) have detailed physics-based health diagnostics available.",
            'action': 'Review detailed diagnostics for optimization',
            'business_impact': 'Opportunity - Enhanced maintenance planning'
        })
    
    # Fleet health summary
    avg_health = np.mean([v['current_soh'] for v in fleet_data.values()])
    if avg_health < 75:
        alerts.append({
            'priority': 'WARNING',
            'title': 'Fleet Health Below Target',
            'message': f"Average fleet health is {avg_health:.1f}%. Consider fleet renewal strategy.",
            'action': 'Develop fleet renewal plan',
            'business_impact': 'Strategic - Fleet performance impact'
        })
    
    return alerts

# --- Main Header ---
st.markdown("""
<div class="fleet-header">
    <h1>🚗⚡ Fleet Battery Intelligence Platform</h1>
    <p><strong>Physics-Informed Dual-Model Framework</strong> | Real-time SOC + Diagnostic SOH + ML Forecasting</p>
    <p><em>Actionable Business Intelligence for Smart Fleet Management</em></p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Fleet Configuration")
    
    # Fleet Management Section
    st.markdown('<div class="diagnostic-panel">', unsafe_allow_html=True)
    st.subheader("🚗 Fleet Management")
    
    fleet_size = st.slider("Fleet Size", min_value=5, max_value=25, value=15, 
                          help="Number of vehicles in your fleet")
    
    if st.button("🔄 Initialize/Refresh Fleet Data", use_container_width=True):
        with st.spinner("Generating physics-informed fleet data..."):
            st.session_state.fleet_data = generate_fleet_data(fleet_size)
            st.session_state.fleet_initialized = True
        st.success(f"Fleet data generated for {fleet_size} vehicles!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis Settings
    st.markdown('<div class="diagnostic-panel">', unsafe_allow_html=True)
    st.subheader("🎯 Analysis Settings")
    
    eol_threshold = st.slider("EOL Threshold (%)", min_value=70, max_value=90, value=80)
    
    analysis_mode = st.selectbox(
        "Analysis Framework",
        ["Dual-Model Physics", "Traditional ML Only", "Real-time Only"],
        help="Select the analysis framework to use"
    )
    
    update_frequency = st.selectbox(
        "Data Update Frequency",
        ["Real-time", "Every 5 minutes", "Every 15 minutes", "Manual"],
        help="How often to refresh fleet data"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Real-time Monitoring
    st.markdown('<div class="diagnostic-panel">', unsafe_allow_html=True)
    st.subheader("📡 Real-time Monitoring")
    
    enable_realtime = st.checkbox("Enable Real-time Updates", value=False)
    show_physics_details = st.checkbox("Show Physics Diagnostics", value=True)
    enable_alerts = st.checkbox("Enable Smart Alerts", value=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- Initialize Fleet Data if not done ---
if not st.session_state.fleet_initialized:
    with st.spinner("Initializing Fleet Battery Intelligence Platform..."):
        st.session_state.fleet_data = generate_fleet_data(fleet_size)
        st.session_state.fleet_initialized = True

# --- Main Dashboard ---
if st.session_state.fleet_initialized:
    fleet_data = st.session_state.fleet_data
    
    # --- Fleet Command Center ---
    st.header("📊 Fleet Command Center", divider='blue')
    
    # Fleet Overview Statistics
    total_vehicles = len(fleet_data)
    avg_health = np.mean([v['current_soh'] for v in fleet_data.values()])
    critical_count = len([v for v in fleet_data.values() if v['rul_prediction'] and v['rul_prediction'] <= 20])
    physics_count = len([v for v in fleet_data.values() if v['physics_enhanced']])
    
    stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)
    
    with stat_col1:
        st.markdown('<div class="realtime-metric">', unsafe_allow_html=True)
        st.metric("🚗 Total Fleet", f"{total_vehicles}", help="Active vehicles in fleet")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with stat_col2:
        health_status = "Excellent" if avg_health > 90 else "Good" if avg_health > 80 else "Fair"
        st.markdown('<div class="realtime-metric">', unsafe_allow_html=True)
        st.metric("💚 Avg Health", f"{avg_health:.1f}%", delta=health_status)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with stat_col3:
        st.markdown('<div class="realtime-metric">', unsafe_allow_html=True)
        st.metric("⚠️ Needs Attention", f"{critical_count}", help="Vehicles with RUL ≤ 20 cycles")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with stat_col4:
        st.markdown('<div class="realtime-metric">', unsafe_allow_html=True)
        st.metric("🧬 Physics Enhanced", f"{physics_count}", help="Vehicles with physics diagnostics")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with stat_col5:
        st.markdown('<div class="realtime-metric">', unsafe_allow_html=True)
        operational_count = len([v for v in fleet_data.values() if v['current_soh'] > 80])
        st.metric("✅ Operational", f"{operational_count}", help="Vehicles above EOL threshold")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # --- Smart Fleet Alerts ---
    if enable_alerts:
        st.header("🚨 Smart Fleet Alerts", divider='red')
        
        fleet_alerts = generate_fleet_alerts(fleet_data)
        
        if fleet_alerts:
            for alert in fleet_alerts:
                if alert['priority'] == 'CRITICAL':
                    st.markdown(f'<div class="alert-critical">', unsafe_allow_html=True)
                    st.error(f"🚨 **{alert['title']}**\n\n{alert['message']}\n\n**Action:** {alert['action']}\n\n**Business Impact:** {alert['business_impact']}")
                    st.markdown('</div>', unsafe_allow_html=True)
                elif alert['priority'] == 'WARNING':
                    st.warning(f"⚠️ **{alert['title']}**\n\n{alert['message']}\n\n**Action:** {alert['action']}\n\n**Business Impact:** {alert['business_impact']}")
                else:
                    st.info(f"ℹ️ **{alert['title']}**\n\n{alert['message']}\n\n**Action:** {alert['action']}\n\n**Business Impact:** {alert['business_impact']}")
        else:
            st.success("✅ **All Systems Operational** - No immediate alerts for your fleet.")
    
    # --- Fleet Vehicle Grid ---
    st.header("🚗 Fleet Vehicle Status", divider='green')
    
    # Sort vehicles by priority (critical first)
    sorted_vehicles = sorted(fleet_data.items(), 
                           key=lambda x: (x[1]['rul_prediction'] if x[1]['rul_prediction'] else 999, -x[1]['current_soh']))
    
    # Create vehicle grid (3 columns)
    cols = st.columns(3)
    
    for idx, (vehicle_id, vehicle_data) in enumerate(sorted_vehicles):
        col_idx = idx % 3
        
        with cols[col_idx]:
            # Determine status class
            health_status = vehicle_data['health_status'].lower()
            status_class = f"status-{health_status}"
            
            st.markdown(f'<div class="vehicle-card {status_class}">', unsafe_allow_html=True)
            
            # Vehicle header
            st.markdown(f"**🚗 {vehicle_id}** | {vehicle_data['vehicle_type']}")
            st.markdown(f"📍 Location: {vehicle_data['location']}")
            
            # Key metrics
            rul_display = f"{vehicle_data['rul_prediction']:.0f} cycles" if vehicle_data['rul_prediction'] else "N/A"
            
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("Health", f"{vehicle_data['current_soh']:.1f}%")
            with metric_col2:
                st.metric("RUL", rul_display)
            
            # Physics enhancement indicator
            if vehicle_data['physics_enhanced']:
                st.markdown("🧬 **Physics-Enhanced Analysis**")
            
            # Method and confidence
            st.markdown(f"**Method:** {vehicle_data['rul_method']}")
            st.markdown(f"**Confidence:** {vehicle_data['rul_confidence']:.0f}%")
            
            # Anomaly indicator
            if vehicle_data['anomaly_count'] > 0:
                st.markdown(f"⚠️ **{vehicle_data['anomaly_count']} anomalies detected**")
            
            # Action button
            if st.button(f"🔍 Detailed Analysis", key=f"detail_{vehicle_id}", use_container_width=True):
                st.session_state.selected_vehicle = vehicle_id
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # --- Detailed Vehicle Analysis ---
    if st.session_state.selected_vehicle:
        selected_id = st.session_state.selected_vehicle
        selected_data = fleet_data[selected_id]
        
        st.header(f"🔬 Detailed Analysis: {selected_id}", divider='purple')
        
        # Close button
        if st.button("← Back to Fleet Overview"):
            st.session_state.selected_vehicle = None
            st.rerun()
        
        # Create tabs for detailed analysis
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Health Overview", 
            "🧬 Physics Diagnostics", 
            "📈 Trend Analysis",
            "🚨 Anomaly Report",
            "💡 Recommendations"
        ])
        
        with tab1:
            st.subheader(f"Health Overview - {selected_id}")
            
            # Real-time metrics
            realtime = selected_data['realtime_state']
            
            rt_col1, rt_col2, rt_col3, rt_col4 = st.columns(4)
            
            with rt_col1:
                st.markdown('<div class="realtime-metric">', unsafe_allow_html=True)
                st.metric("Real-time SOC", f"{realtime['realtime_soc']:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with rt_col2:
                st.markdown('<div class="realtime-metric">', unsafe_allow_html=True)
                st.metric("Real-time SOH", f"{realtime['realtime_soh']:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with rt_col3:
                st.markdown('<div class="realtime-metric">', unsafe_allow_html=True)
                st.metric("OCV", f"{realtime['ocv']:.3f}V")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with rt_col4:
                temp_corrected = "✅" if realtime['temperature_corrected'] else "❌"
                st.markdown('<div class="realtime-metric">', unsafe_allow_html=True)
                st.metric("Temp Corrected", temp_corrected)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Historical trend chart
            df = selected_data['historical_data']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['cycle'], 
                y=df['soh_percent'],
                mode='lines+markers',
                name='State of Health',
                line=dict(color='#1f77b4', width=3)
            ))
            
            fig.add_hline(y=eol_threshold, line_dash="dash", line_color="red",
                         annotation_text=f"EOL Threshold ({eol_threshold}%)")
            
            fig.update_layout(
                title="State of Health Trend",
                xaxis_title="Cycle",
                yaxis_title="SOH (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            if show_physics_details and selected_data['physics_enhanced']:
                st.subheader("🧬 Physics-Informed Diagnostics")
                
                physics_analysis = selected_data['physics_soh_analysis']
                
                if physics_analysis and 'soh_estimate' in physics_analysis:
                    st.markdown('<div class="physics-insight">', unsafe_allow_html=True)
                    st.write(f"**Physics SOH Estimate:** {physics_analysis['soh_estimate']:.1f}%")
                    st.write(f"**Analysis Quality:** {physics_analysis.get('analysis_quality', 'N/A')}")
                    
                    if 'peaks' in physics_analysis and physics_analysis['peaks']:
                        st.write(f"**ICA Peaks Detected:** {len(physics_analysis['peaks'])}")
                        
                        # Display peak information
                        for i, peak in enumerate(physics_analysis['peaks'][:3]):  # Show first 3 peaks
                            peak_info = list(peak.values())
                            if len(peak_info) >= 3:
                                st.write(f"Peak {i+1}: Voltage={peak_info[0]:.3f}V, SOC={peak_info[1]:.3f}, Height={peak_info[2]:.2f}")
                    
                    if 'health_indicators' in physics_analysis:
                        indicators = physics_analysis['health_indicators']
                        st.write(f"**Signal Strength:** {indicators.get('signal_strength', 'N/A'):.2f}")
                        st.write(f"**Peak Symmetry:** {indicators.get('peak_symmetry', 'N/A'):.3f}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # ICA Curve Visualization
                    if 'ica_curve' in physics_analysis:
                        ica_data = physics_analysis['ica_curve']
                        
                        fig_ica = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=['Open Circuit Voltage vs SOC', 'Incremental Capacity Analysis (ICA)']
                        )
                        
                        # OCV curve
                        fig_ica.add_trace(
                            go.Scatter(x=ica_data['soc'], y=ica_data['voltage'], 
                                     mode='lines', name='OCV Curve', line=dict(color='blue')),
                            row=1, col=1
                        )
                        
                        # ICA curve
                        fig_ica.add_trace(
                            go.Scatter(x=ica_data['voltage'], y=ica_data['dq_dv'], 
                                     mode='lines', name='dQ/dV', line=dict(color='red')),
                            row=2, col=1
                        )
                        
                        fig_ica.update_layout(height=600, title_text="Physics-Based Battery Analysis")
                        fig_ica.update_xaxes(title_text="SOC", row=1, col=1)
                        fig_ica.update_yaxes(title_text="Voltage (V)", row=1, col=1)
                        fig_ica.update_xaxes(title_text="Voltage (V)", row=2, col=1)
                        fig_ica.update_yaxes(title_text="dQ/dV (Ah/V)", row=2, col=1)
                        
                        st.plotly_chart(fig_ica, use_container_width=True)
                
                else:
                    st.warning("Physics diagnostics not available for this vehicle")
            else:
                st.info("Physics diagnostics disabled or not available for this vehicle")
        
        with tab3:
            st.subheader("📈 Comprehensive Trend Analysis")
            
            df = selected_data['historical_data']
            
            # Multi-parameter visualization
            fig_trends = make_subplots(
                rows=2, cols=2,
                subplot_titles=['SOH Trend', 'Capacity Trend', 'Voltage Profile', 'Temperature Profile']
            )
            
            # SOH trend
            fig_trends.add_trace(
                go.Scatter(x=df['cycle'], y=df['soh_percent'], mode='lines+markers', 
                          name='SOH', line=dict(color='blue')), row=1, col=1
            )
            
            # Capacity trend
            if 'capacity_ah' in df.columns:
                fig_trends.add_trace(
                    go.Scatter(x=df['cycle'], y=df['capacity_ah'], mode='lines', 
                              name='Capacity', line=dict(color='green')), row=1, col=2
                )
            
            # Voltage profile
            if 'avg_voltage' in df.columns:
                fig_trends.add_trace(
                    go.Scatter(x=df['cycle'], y=df['avg_voltage'], mode='lines', 
                              name='Voltage', line=dict(color='orange')), row=2, col=1
                )
            
            # Temperature profile
            if 'avg_temp_c' in df.columns:
                fig_trends.add_trace(
                    go.Scatter(x=df['cycle'], y=df['avg_temp_c'], mode='lines', 
                              name='Temperature', line=dict(color='red')), row=2, col=2
                )
            
            fig_trends.update_layout(height=600, showlegend=False, title_text="Multi-Parameter Trend Analysis")
            st.plotly_chart(fig_trends, use_container_width=True)
            
            # Degradation metrics
            st.subheader("📊 Degradation Metrics")
            
            initial_soh = df['soh_percent'].iloc[0]
            current_soh = df['soh_percent'].iloc[-1]
            total_cycles = len(df)
            degradation_rate = (initial_soh - current_soh) / total_cycles
            
            deg_col1, deg_col2, deg_col3 = st.columns(3)
            
            with deg_col1:
                st.metric("Total Degradation", f"{initial_soh - current_soh:.1f}%")
            with deg_col2:
                st.metric("Degradation Rate", f"{degradation_rate:.4f}%/cycle")
            with deg_col3:
                projected_eol = (current_soh - eol_threshold) / degradation_rate if degradation_rate > 0 else float('inf')
                st.metric("Linear EOL Projection", f"{projected_eol:.0f} cycles" if projected_eol != float('inf') else "N/A")
        
        with tab4:
            st.subheader("🚨 Anomaly Detection Report")
            
            analysis = selected_data['comprehensive_analysis']
            anomalies = analysis['anomalies']
            
            if anomalies['count'] > 0:
                st.warning(f"⚠️ {anomalies['count']} anomalies detected in vehicle operation")
                
                # Anomaly summary
                if 'summary' in anomalies:
                    st.write("**Anomaly Types:**")
                    for anomaly_type, count in anomalies['summary'].items():
                        st.write(f"- {anomaly_type}: {count}")
                
                # Anomaly visualization
                df = selected_data['historical_data']
                anomaly_fig = go.Figure()
                
                # Normal operation line
                anomaly_fig.add_trace(go.Scatter(
                    x=df['cycle'], 
                    y=df['soh_percent'],
                    mode='lines',
                    name='Normal Operation',
                    line=dict(color='blue', width=2)
                ))
                
                # Mark anomaly points if available
                if 'cycles' in anomalies and anomalies['cycles']:
                    anomaly_cycles = anomalies['cycles']
                    anomaly_soh_values = []
                    
                    for cycle in anomaly_cycles:
                        matching_rows = df[df['cycle'] == cycle]
                        if not matching_rows.empty:
                            anomaly_soh_values.append(matching_rows['soh_percent'].iloc[0])
                        else:
                            anomaly_soh_values.append(df['soh_percent'].mean())
                    
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
                
            else:
                st.success("✅ No anomalies detected - Vehicle operating normally")
        
        with tab5:
            st.subheader("💡 Smart Recommendations")
            
            # Generate vehicle-specific recommendations
            vehicle_recommendations = []
            
            rul = selected_data['rul_prediction']
            current_soh = selected_data['current_soh']
            anomaly_count = selected_data['anomaly_count']
            health_score = selected_data['health_score']
            
            # RUL-based recommendations
            if rul and rul <= 5:
                vehicle_recommendations.append({
                    'priority': 'CRITICAL',
                    'category': 'Immediate Action',
                    'recommendation': 'Replace battery immediately - less than 5 cycles remaining',
                    'timeline': 'Within 24 hours',
                    'cost_impact': 'High - Emergency replacement costs'
                })
            elif rul and rul <= 20:
                vehicle_recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Maintenance Planning',
                    'recommendation': 'Schedule battery replacement within next maintenance window',
                    'timeline': 'Within 2 weeks',
                    'cost_impact': 'Medium - Planned replacement costs'
                })
            elif rul and rul <= 50:
                vehicle_recommendations.append({
                    'priority': 'MEDIUM',
                    'category': 'Procurement',
                    'recommendation': 'Order replacement battery and plan installation',
                    'timeline': 'Within 1 month',
                    'cost_impact': 'Medium - Standard replacement costs'
                })
            
            # Health-based recommendations
            if health_score < 60:
                vehicle_recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Diagnostics',
                    'recommendation': 'Perform detailed battery diagnostics and health assessment',
                    'timeline': 'Within 1 week',
                    'cost_impact': 'Low - Diagnostic costs only'
                })
            
            # Anomaly-based recommendations
            if anomaly_count > 15:
                vehicle_recommendations.append({
                    'priority': 'MEDIUM',
                    'category': 'Investigation',
                    'recommendation': 'Investigate operating conditions and driver behavior patterns',
                    'timeline': 'Within 2 weeks',
                    'cost_impact': 'Low - Analysis and training costs'
                })
            
            # Physics enhancement opportunities
            if not selected_data['physics_enhanced']:
                vehicle_recommendations.append({
                    'priority': 'LOW',
                    'category': 'Optimization',
                    'recommendation': 'Collect additional OCV data for enhanced physics diagnostics',
                    'timeline': 'Ongoing',
                    'cost_impact': 'None - Data collection improvement'
                })
            
            # Display recommendations
            if vehicle_recommendations:
                for rec in vehicle_recommendations:
                    if rec['priority'] == 'CRITICAL':
                        st.error(f"🚨 **{rec['category']}** (Priority: {rec['priority']})\n\n"
                                f"**Recommendation:** {rec['recommendation']}\n\n"
                                f"**Timeline:** {rec['timeline']}\n\n"
                                f"**Cost Impact:** {rec['cost_impact']}")
                    elif rec['priority'] == 'HIGH':
                        st.warning(f"⚠️ **{rec['category']}** (Priority: {rec['priority']})\n\n"
                                  f"**Recommendation:** {rec['recommendation']}\n\n"
                                  f"**Timeline:** {rec['timeline']}\n\n"
                                  f"**Cost Impact:** {rec['cost_impact']}")
                    elif rec['priority'] == 'MEDIUM':
                        st.info(f"🔧 **{rec['category']}** (Priority: {rec['priority']})\n\n"
                               f"**Recommendation:** {rec['recommendation']}\n\n"
                               f"**Timeline:** {rec['timeline']}\n\n"
                               f"**Cost Impact:** {rec['cost_impact']}")
                    else:
                        st.success(f"💡 **{rec['category']}** (Priority: {rec['priority']})\n\n"
                                  f"**Recommendation:** {rec['recommendation']}\n\n"
                                  f"**Timeline:** {rec['timeline']}\n\n"
                                  f"**Cost Impact:** {rec['cost_impact']}")
            else:
                st.success("✅ **Vehicle Operating Optimally** - Continue normal monitoring schedule")
            
            # Business impact summary
            st.subheader("📊 Business Impact Summary")
            
            # Calculate estimated costs
            replacement_cost = 15000  # Typical EV battery replacement cost
            diagnostic_cost = 500
            maintenance_cost = 1000
            
            if rul and rul <= 20:
                estimated_cost = replacement_cost
                impact_level = "High"
                impact_description = "Immediate battery replacement required"
            elif rul and rul <= 50:
                estimated_cost = replacement_cost
                impact_level = "Medium"
                impact_description = "Planned battery replacement needed"
            else:
                estimated_cost = maintenance_cost
                impact_level = "Low"
                impact_description = "Continue normal maintenance"
            
            impact_col1, impact_col2, impact_col3 = st.columns(3)
            
            with impact_col1:
                st.metric("Estimated Cost", f"${estimated_cost:,}")
            with impact_col2:
                st.metric("Impact Level", impact_level)
            with impact_col3:
                downtime_hours = 8 if rul and rul <= 20 else 4 if rul and rul <= 50 else 0
                st.metric("Est. Downtime", f"{downtime_hours} hours")
    
    # --- Fleet Analytics Dashboard ---
    st.header("📈 Fleet Analytics Dashboard", divider='orange')
    
    # Create fleet-wide analytics
    analytics_tab1, analytics_tab2, analytics_tab3 = st.tabs([
        "📊 Fleet Health Distribution",
        "🎯 Predictive Maintenance",
        "💰 Cost Analysis"
    ])
    
    with analytics_tab1:
        st.subheader("Fleet Health Distribution Analysis")
        
        # Health status distribution
        health_statuses = [v['health_status'] for v in fleet_data.values()]
        status_counts = pd.Series(health_statuses).value_counts()
        
        fig_health_dist = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Fleet Health Status Distribution",
            color_discrete_map={
                'Excellent': '#28a745',
                'Good': '#20c997',
                'Fair': '#ffc107',
                'Poor': '#fd7e14',
                'Critical': '#dc3545'
            }
        )
        
        st.plotly_chart(fig_health_dist, use_container_width=True)
        
        # SOH distribution histogram
        soh_values = [v['current_soh'] for v in fleet_data.values()]
        
        fig_soh_dist = px.histogram(
            x=soh_values,
            nbins=20,
            title="State of Health Distribution Across Fleet",
            labels={'x': 'State of Health (%)', 'y': 'Number of Vehicles'}
        )
        fig_soh_dist.add_vline(x=eol_threshold, line_dash="dash", line_color="red",
                               annotation_text=f"EOL Threshold ({eol_threshold}%)")
        
        st.plotly_chart(fig_soh_dist, use_container_width=True)
    
    with analytics_tab2:
        st.subheader("Predictive Maintenance Planning")
        
        # RUL distribution
        rul_values = [v['rul_prediction'] for v in fleet_data.values() if v['rul_prediction']]
        
        if rul_values:
            fig_rul_dist = px.histogram(
                x=rul_values,
                nbins=15,
                title="Remaining Useful Life Distribution",
                labels={'x': 'RUL (cycles)', 'y': 'Number of Vehicles'}
            )
            fig_rul_dist.add_vline(x=20, line_dash="dash", line_color="orange",
                                   annotation_text="Attention Threshold (20 cycles)")
            fig_rul_dist.add_vline(x=5, line_dash="dash", line_color="red",
                                   annotation_text="Critical Threshold (5 cycles)")
            
            st.plotly_chart(fig_rul_dist, use_container_width=True)
            
            # Maintenance timeline
            st.subheader("📅 Maintenance Timeline")
            
            maintenance_schedule = []
            for vehicle_id, data in fleet_data.items():
                if data['rul_prediction'] and data['rul_prediction'] <= 100:
                    estimated_date = datetime.now() + timedelta(days=data['rul_prediction'] * 7)  # Assume 1 cycle = 1 week
                    maintenance_schedule.append({
                        'Vehicle': vehicle_id,
                        'Current SOH': f"{data['current_soh']:.1f}%",
                        'RUL (cycles)': data['rul_prediction'],
                        'Est. Replacement Date': estimated_date.strftime('%Y-%m-%d'),
                        'Priority': 'Critical' if data['rul_prediction'] <= 10 else 'High' if data['rul_prediction'] <= 30 else 'Medium'
                    })
            
            if maintenance_schedule:
                maintenance_df = pd.DataFrame(maintenance_schedule)
                maintenance_df = maintenance_df.sort_values('RUL (cycles)')
                st.dataframe(maintenance_df, use_container_width=True)
            else:
                st.info("No vehicles require immediate maintenance scheduling")
    
    with analytics_tab3:
        st.subheader("💰 Fleet Cost Analysis")
        
        # Calculate fleet costs
        total_vehicles = len(fleet_data)
        vehicles_needing_replacement = len([v for v in fleet_data.values() if v['rul_prediction'] and v['rul_prediction'] <= 50])
        vehicles_critical = len([v for v in fleet_data.values() if v['rul_prediction'] and v['rul_prediction'] <= 10])
        
        # Cost estimates
        battery_cost = 15000
        emergency_multiplier = 1.5
        diagnostic_cost = 500
        
        planned_replacement_cost = (vehicles_needing_replacement - vehicles_critical) * battery_cost
        emergency_replacement_cost = vehicles_critical * battery_cost * emergency_multiplier
        diagnostic_costs = total_vehicles * diagnostic_cost
        
        total_estimated_cost = planned_replacement_cost + emergency_replacement_cost + diagnostic_costs
        
        cost_col1, cost_col2, cost_col3, cost_col4 = st.columns(4)
        
        with cost_col1:
            st.metric("Planned Replacements", f"${planned_replacement_cost:,}")
        with cost_col2:
            st.metric("Emergency Replacements", f"${emergency_replacement_cost:,}")
        with cost_col3:
            st.metric("Diagnostics Budget", f"${diagnostic_costs:,}")
        with cost_col4:
            st.metric("Total Estimated", f"${total_estimated_cost:,}")
        
        # Cost breakdown chart
        cost_breakdown = {
            'Cost Category': ['Planned Replacements', 'Emergency Replacements', 'Diagnostics', 'Other Maintenance'],
            'Amount': [planned_replacement_cost, emergency_replacement_cost, diagnostic_costs, total_vehicles * 2000]
        }
        
        cost_df = pd.DataFrame(cost_breakdown)
        
        fig_costs = px.bar(
            cost_df,
            x='Cost Category',
            y='Amount',
            title="Estimated Fleet Maintenance Costs",
            labels={'Amount': 'Cost ($)'}
        )
        
        st.plotly_chart(fig_costs, use_container_width=True)
        
        # ROI Analysis
        st.subheader("📊 ROI of Predictive Maintenance")
        
        # Calculate savings from predictive vs reactive maintenance
        reactive_emergency_rate = 0.3  # 30% of failures would be emergency without prediction
        emergency_premium = 0.5  # 50% cost premium for emergency repairs
        downtime_cost_per_hour = 200
        
        total_potential_failures = vehicles_needing_replacement
        avoided_emergencies = int(total_potential_failures * reactive_emergency_rate)
        cost_savings = avoided_emergencies * battery_cost * emergency_premium
        downtime_savings = avoided_emergencies * 8 * downtime_cost_per_hour  # 8 hours average emergency downtime
        
        roi_col1, roi_col2, roi_col3 = st.columns(3)
        
        with roi_col1:
            st.metric("Avoided Emergencies", f"{avoided_emergencies}")
        with roi_col2:
            st.metric("Cost Savings", f"${cost_savings:,}")
        with roi_col3:
            st.metric("Downtime Savings", f"${downtime_savings:,}")
        
        total_savings = cost_savings + downtime_savings
        platform_cost = 50000  # Estimated annual platform cost
        roi_percentage = ((total_savings - platform_cost) / platform_cost) * 100 if platform_cost > 0 else 0
        
        st.success(f"🎯 **Annual ROI: {roi_percentage:.1f}%** (${total_savings:,} savings vs ${platform_cost:,} platform cost)")

# --- Real-time Updates ---
if enable_realtime and st.session_state.fleet_initialized:
    # Auto-refresh every 30 seconds in real-time mode
    time.sleep(0.1)  # Small delay to prevent excessive reloading
    
    # Update timestamp
    st.sidebar.success(f"🔄 Last updated: {datetime.now().strftime('%H:%M:%S')}")
    
    # Auto-refresh button
    if st.sidebar.button("🔄 Force Refresh", use_container_width=True):
        st.rerun()

# --- Export and Integration ---
st.header("📤 Export & Integration", divider='gray')

export_col1, export_col2, export_col3, export_col4 = st.columns(4)

with export_col1:
    if st.button("📊 Export Fleet Report", use_container_width=True):
        # Create summary report
        if st.session_state.fleet_initialized:
            report_data = {
                'Fleet Summary': {
                    'Total Vehicles': len(fleet_data),
                    'Average Health': f"{np.mean([v['current_soh'] for v in fleet_data.values()]):.1f}%",
                    'Vehicles Needing Attention': len([v for v in fleet_data.values() if v['rul_prediction'] and v['rul_prediction'] <= 20]),
                    'Physics Enhanced': len([v for v in fleet_data.values() if v['physics_enhanced']])
                }
            }
            
            st.download_button(
                label="Download Report (JSON)",
                data=json.dumps(report_data, indent=2, default=str),
                file_name=f"fleet_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

with export_col2:
    if st.button("📋 Export Vehicle Data", use_container_width=True):
        if st.session_state.fleet_initialized:
            # Create vehicle summary
            vehicle_summary = []
            for vehicle_id, data in fleet_data.items():
                vehicle_summary.append({
                    'Vehicle_ID': vehicle_id,
                    'Vehicle_Type': data['vehicle_type'],
                    'Location': data['location'],
                    'Current_SOH': data['current_soh'],
                    'Health_Status': data['health_status'],
                    'RUL_Prediction': data['rul_prediction'],
                    'Confidence': data['rul_confidence'],
                    'Physics_Enhanced': data['physics_enhanced'],
                    'Anomaly_Count': data['anomaly_count']
                })
            
            summary_df = pd.DataFrame(vehicle_summary)
            csv = summary_df.to_csv(index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"fleet_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

with export_col3:
    if st.button("🔗 API Integration", use_container_width=True):
        st.info("""
        **API Endpoints Available:**
        - GET /api/fleet/status
        - GET /api/vehicle/{id}/health
        - GET /api/fleet/alerts
        - POST /api/vehicle/{id}/data
        - GET /api/analytics/costs
        """)

with export_col4:
    if st.button("⚙️ System Settings", use_container_width=True):
        st.info("""
        **System Configuration:**
        - Real-time data ingestion
        - Automated alert thresholds
        - Maintenance scheduling
        - Cost optimization settings
        """)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>🚗⚡ Fleet Battery Intelligence Platform</strong> | Physics-Informed Dual-Model Framework</p>
    <p><em>Real-time SOC (Hu et al.) • Diagnostic SOH (Weng et al.) • ML Forecasting • Business Intelligence</em></p>
    <p>Transforming Fleet Management with Actionable Battery Intelligence</p>
</div>
""", unsafe_allow_html=True)