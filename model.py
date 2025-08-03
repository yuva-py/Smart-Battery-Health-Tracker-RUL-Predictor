import pandas as pd
import numpy as np
import joblib
import scipy.io
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.cluster import DBSCAN
from scipy.signal import savgol_filter
from scipy.stats import linregress, zscore
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

def get_cycle_type(cycle_struct):
        """Safely extracts the cycle type string from a nested structure."""
        try:
            # Start with the most likely, deeply nested structure
            return cycle_struct['type'][0][0][0].strip()
        except (IndexError, KeyError):
            # If that fails, try less nested structures
            try:
                return cycle_struct['type'][0].strip()
            except (IndexError, KeyError):
                # If all else fails, return an unknown type
                return "unknown"
class AdvancedBatteryRULPredictor:
    """
    Enhanced Battery RUL Predictor with improved predictions for batteries near EOL
    and better early-stage predictions for maintenance planning.
    """
    def __init__(self, eol_threshold=80, battery_type="Li-ion 18650"):
        self.eol_threshold = eol_threshold
        self.battery_type = battery_type
        self.models = {}
        self.scaler = StandardScaler()
        self.anomaly_detector = None
        
        # IMPROVED: More nuanced health categories
        self.health_categories = {
            'excellent': (95, 100),
            'good': (85, 95),
            'fair': (75, 85),
            'poor': (65, 75),
            'critical': (50, 65),
            'eol': (0, 50)
        }
        
        self.battery_specs = {
            "Li-ion 18650": {"nominal_capacity": 2.5, "nominal_voltage": 3.7, "cycle_life": 500},
            "Li-ion Pouch": {"nominal_capacity": 20.0, "nominal_voltage": 3.7, "cycle_life": 1000},
            "LiFePO4": {"nominal_capacity": 10.0, "nominal_voltage": 3.2, "cycle_life": 2000},
            "Li-polymer": {"nominal_capacity": 5.0, "nominal_voltage": 3.7, "cycle_life": 300}
        }
    # In app.py, after your imports

    
    def get_health_status(self, soh_percent):
        """Improved health status with more categories"""
        if soh_percent is None or not isinstance(soh_percent, (int, float)):
            return "Invalid Input"
        
        soh_percent = max(0, min(soh_percent, 100))
        
        for status, (min_val, max_val) in self.health_categories.items():
            if min_val <= soh_percent < max_val:
                return status.title()
        
        return "Excellent" if soh_percent >= 95 else "Unknown"

    def calculate_health_score(self, df):
        """Enhanced health score calculation"""
        current_soh = df['soh_percent'].iloc[-1]
        soh_score = current_soh
        
        # Degradation rate analysis (improved)
        if len(df) >= 10:
            recent_cycles = df['cycle'].tail(10).values
            recent_soh = df['soh_percent'].tail(10).values
            
            if len(recent_cycles) > 1:
                degradation_rate = abs(np.polyfit(recent_cycles, recent_soh, 1)[0])
                deg_score = max(0, 100 - (degradation_rate * 5000))
            else:
                deg_score = 70
        else:
            deg_score = 70
        
        # Voltage stability (if available)
        if 'avg_voltage' in df.columns:
            voltage_std = df['avg_voltage'].std()
            voltage_score = max(0, 100 - (voltage_std * 200))
        else:
            voltage_score = 70
        
        # Temperature stability (if available)
        if 'avg_temp_c' in df.columns:
            temp_range = df['avg_temp_c'].max() - df['avg_temp_c'].min()
            temp_score = max(0, 100 - (temp_range * 1.5))
        else:
            temp_score = 70
        
        # Capacity consistency (if available)
        if 'capacity_ah' in df.columns and len(df) >= 5:
            capacity_std = df['capacity_ah'].std()
            capacity_mean = df['capacity_ah'].mean()
            capacity_cv = capacity_std / capacity_mean if capacity_mean > 0 else 1
            capacity_score = max(0, 100 - (capacity_cv * 500))
        else:
            capacity_score = 70
        
        # Weighted health score
        health_score = (soh_score * 0.4 + deg_score * 0.25 + 
                       voltage_score * 0.15 + temp_score * 0.1 + capacity_score * 0.1)
        
        return min(100, max(0, health_score))

    def create_advanced_features(self, df):
        """Enhanced feature engineering"""
        df_enhanced = df.copy()
        
        # Rolling window features
        for window in [3, 5, 10, 20]:
            if len(df) >= window:
                df_enhanced[f'soh_rolling_mean_{window}'] = df['soh_percent'].rolling(window=window).mean()
                df_enhanced[f'soh_rolling_std_{window}'] = df['soh_percent'].rolling(window=window).std()
                
                if 'avg_voltage' in df.columns:
                    df_enhanced[f'voltage_rolling_mean_{window}'] = df['avg_voltage'].rolling(window=window).mean()
                if 'avg_temp_c' in df.columns:
                    df_enhanced[f'temp_rolling_mean_{window}'] = df['avg_temp_c'].rolling(window=window).mean()
                if 'capacity_ah' in df.columns:
                    df_enhanced[f'capacity_rolling_mean_{window}'] = df['capacity_ah'].rolling(window=window).mean()
        
        # Lag features
        for lag in [1, 5, 10]:
            df_enhanced[f'soh_diff_{lag}'] = df['soh_percent'].diff(lag)
            if 'avg_voltage' in df.columns:
                df_enhanced[f'voltage_diff_{lag}'] = df['avg_voltage'].diff(lag)
            if 'capacity_ah' in df.columns:
                df_enhanced[f'capacity_diff_{lag}'] = df['capacity_ah'].diff(lag)
        
        # Advanced derived features
        df_enhanced['cumulative_degradation'] = 100 - df['soh_percent']
        df_enhanced['degradation_acceleration'] = df_enhanced['soh_diff_1'].diff(1)
        
        if 'avg_voltage' in df.columns:
            df_enhanced['voltage_drop'] = df['avg_voltage'].iloc[0] - df['avg_voltage']
        if 'capacity_ah' in df.columns:
            df_enhanced['capacity_loss'] = df['capacity_ah'].iloc[0] - df['capacity_ah']
        
        # Polynomial features
        df_enhanced['cycle_squared'] = df['cycle'] ** 2
        df_enhanced['cycle_log'] = np.log(df['cycle'] + 1)
        df_enhanced['cycle_sqrt'] = np.sqrt(df['cycle'])
        
        # IMPROVED: Better trend analysis
        for i in range(len(df_enhanced)):
            if i >= 10:
                recent_cycles = df_enhanced['cycle'].iloc[i-9:i+1].values
                recent_soh = df_enhanced['soh_percent'].iloc[i-9:i+1].values
                if len(recent_cycles) > 1:
                    slope, _, r_value, _, _ = linregress(recent_cycles, recent_soh)
                    df_enhanced.loc[df_enhanced.index[i], 'recent_trend_slope'] = slope
                    df_enhanced.loc[df_enhanced.index[i], 'recent_trend_r2'] = r_value ** 2
                    df_enhanced.loc[df_enhanced.index[i], 'trend_strength'] = abs(slope) * r_value ** 2
        
        # Smoothed features
        if len(df_enhanced) > 5:
            window_length = min(11, len(df_enhanced) if len(df_enhanced) % 2 == 1 else len(df_enhanced) - 1)
            if window_length >= 5:
                try:
                    df_enhanced['soh_smoothed'] = savgol_filter(df['soh_percent'], window_length, 3)
                    if 'avg_voltage' in df.columns:
                        df_enhanced['voltage_smoothed'] = savgol_filter(df['avg_voltage'], window_length, 3)
                    df_enhanced['smoothed_degradation_rate'] = df_enhanced['soh_smoothed'].diff(1)
                except:
                    pass
        
        return df_enhanced

    def detect_anomalies(self, df, method="all"):
        """Improved anomaly detection with better structure"""
        all_anomalies = []
        anomaly_counts = defaultdict(int)
        
        if df.empty or "cycle" not in df.columns:
            return {"anomalies": [], "summary": {}, "count": 0, "cycles": [], "types": []}
        
        timestamp_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        soh_column = 'soh_percent' if 'soh_percent' in df.columns else 'SoH'
        temp_column = 'avg_temp_c' if 'avg_temp_c' in df.columns else 'Temperature'
        
        # Statistical anomalies
        if method in ["statistical", "all"] and soh_column in df.columns:
            soh = df[soh_column]
            mean_soh = soh.mean()
            std_soh = soh.std()
            
            for i, value in enumerate(soh):
                z_score = (value - mean_soh) / std_soh if std_soh > 0 else 0
                if abs(z_score) > 2.5:
                    anomaly = {
                        "cycle": int(df.iloc[i]["cycle"]),
                        "timestamp": timestamp_now,
                        "type": "Statistical",
                        "column": soh_column,
                        "score": round(z_score, 2),
                        "description": f"Z-score anomaly in {soh_column}",
                        "source": "Z-score method"
                    }
                    all_anomalies.append(anomaly)
                    anomaly_counts["Statistical"] += 1
        
        # Pattern anomalies
        if method in ["pattern", "all"] and soh_column in df.columns:
            for i in range(1, len(df)):
                delta = df[soh_column].iloc[i] - df[soh_column].iloc[i - 1]
                if delta > 2.0:  # Unusual increase
                    anomaly = {
                        "cycle": int(df.iloc[i]["cycle"]),
                        "timestamp": timestamp_now,
                        "type": "Pattern",
                        "column": soh_column,
                        "score": round(delta, 2),
                        "description": f"Unusual {soh_column} increase",
                        "source": "Delta threshold"
                    }
                    all_anomalies.append(anomaly)
                    anomaly_counts["Pattern"] += 1
                elif delta < -3.0:  # Sudden drop
                    anomaly = {
                        "cycle": int(df.iloc[i]["cycle"]),
                        "timestamp": timestamp_now,
                        "type": "Pattern",
                        "column": soh_column,
                        "score": round(abs(delta), 2),
                        "description": f"Sudden {soh_column} drop",
                        "source": "Delta threshold"
                    }
                    all_anomalies.append(anomaly)
                    anomaly_counts["Pattern"] += 1
        
        # Create structured output
        cycles = [anomaly['cycle'] for anomaly in all_anomalies]
        types = [anomaly['type'] for anomaly in all_anomalies]
        
        return {
            "anomalies": all_anomalies,
            "summary": dict(anomaly_counts),
            "count": len(all_anomalies),
            "cycles": cycles,
            "types": types
        }

    def estimate_rul_ensemble(self, df, use_models=['trend_analysis', 'linear', 'exponential_smoothing', 'polynomial', 'advanced_trend']):
        """FIXED: Better ensemble with proper EOL handling"""
        predictions = {}
        model_reliability = {}
        
        current_soh = df['soh_percent'].iloc[-1]
        
        print(f"🔍 Current SoH: {current_soh:.1f}%, EOL Threshold: {self.eol_threshold}%")
        
        # FIXED: Better EOL detection and handling
        if current_soh < self.eol_threshold:
            print(f"⚠️ Battery below EOL threshold - using conservative estimation")
            return self._estimate_eol_rul(df), predictions, model_reliability
        
        # Run individual models
        for model_name in use_models:
            if model_name == 'trend_analysis':
                rul = self._estimate_rul_trend_analysis_improved(df)
                predictions['trend_analysis'] = rul
                model_reliability['trend_analysis'] = 0.9 if rul else 0
                
            elif model_name == 'linear':
                rul = self._estimate_rul_linear_improved(df)
                predictions['linear'] = rul
                model_reliability['linear'] = 0.8 if rul else 0
                
            elif model_name == 'exponential_smoothing':
                rul = self._estimate_rul_exponential_smoothing_improved(df)
                predictions['exponential_smoothing'] = rul
                model_reliability['exponential_smoothing'] = 0.85 if rul else 0
                
            elif model_name == 'polynomial':
                rul = self._estimate_rul_polynomial_improved(df)
                predictions['polynomial'] = rul
                model_reliability['polynomial'] = 0.75 if rul else 0
                
            elif model_name == 'advanced_trend':
                rul = self._estimate_rul_advanced_trend(df)
                predictions['advanced_trend'] = rul
                model_reliability['advanced_trend'] = 0.85 if rul else 0
        
        print(f"🔮 Individual predictions: {predictions}")
        
        # Filter valid predictions
        valid_predictions = {k: v for k, v in predictions.items() if v is not None and v > 0}
        
        if not valid_predictions:
            print("❌ No valid predictions from any model")
            return None, predictions, model_reliability
        
        # Improved ensemble combination
        ensemble_rul = self._adaptive_ensemble_improved(valid_predictions, model_reliability, df)
        print(f"🎯 Final ensemble prediction: {ensemble_rul} cycles")
        return ensemble_rul, predictions, model_reliability

    def _estimate_eol_rul(self, df):
        """FIXED: Better handling for batteries at/near EOL"""
        current_soh = df['soh_percent'].iloc[-1]
        
        print(f"🔋 EOL estimation for SoH: {current_soh:.1f}%")
        
        if current_soh <= self.eol_threshold - 10:
            print("🚨 Battery significantly past EOL")
            return 0  # Already significantly past EOL
        elif current_soh <= self.eol_threshold - 5:
            print("⚠️ Battery past EOL threshold")
            return 1  # Past EOL but still functioning
        elif current_soh <= self.eol_threshold:
            print("📍 Battery at EOL threshold")
            return 2  # At EOL threshold, minimal life remaining
        else:
            # Close to EOL, estimate more carefully
            remaining_capacity = current_soh - self.eol_threshold
            print(f"🔍 Remaining capacity before EOL: {remaining_capacity:.1f}%")
            
            if len(df) >= 10:
                # Use recent degradation trend
                recent_data = df.tail(10)
                if len(recent_data) > 1:
                    try:
                        slope, _, r_value, _, _ = linregress(recent_data['cycle'], recent_data['soh_percent'])
                        if slope < 0 and r_value**2 > 0.1:  # Valid degradation trend
                            degradation_per_cycle = abs(slope)
                            estimated_rul = max(3, int(remaining_capacity / degradation_per_cycle))
                            print(f"📊 Trend-based RUL: {estimated_rul} cycles")
                            return min(estimated_rul, 50)  # Cap at 50 cycles for near-EOL
                    except:
                        pass
            
            # Fallback estimation
            conservative_rul = max(3, int(remaining_capacity * 2))  # Conservative estimate
            print(f"🛡️ Conservative RUL estimate: {conservative_rul} cycles")
            return min(conservative_rul, 30)

    def _estimate_rul_trend_analysis_improved(self, df):
        """FIXED: Improved trend analysis with better boundary handling"""
        if len(df) < 10:
            return None
        
        current_soh = df['soh_percent'].iloc[-1]
        
        print(f"📈 Trend analysis for SoH: {current_soh:.1f}%")
        
        # FIXED: Don't immediately return EOL estimation for near-threshold batteries
        if current_soh < self.eol_threshold - 5:
            return self._estimate_eol_rul(df)
        
        # Multiple window analysis for robustness
        best_rul = None
        best_confidence = 0
        
        # FIXED: More adaptive window sizing
        data_length = len(df)
        min_window = max(10, data_length // 10)
        max_window = min(data_length, data_length // 2)
        
        for window_size in range(min_window, max_window + 1, max(1, (max_window - min_window) // 5)):
            if len(df) >= window_size:
                recent_data = df.tail(window_size)
                cycles = recent_data['cycle'].values
                soh_values = recent_data['soh_percent'].values
                
                try:
                    slope, intercept, r_value, _, _ = linregress(cycles, soh_values)
                    r_squared = r_value ** 2
                    
                    print(f"  📊 Window {window_size}: slope={slope:.6f}, R²={r_squared:.3f}")
                    
                    # FIXED: Better validation criteria
                    if (r_squared > 0.15 and slope < -0.001):  # Must have meaningful negative slope
                        # Calculate RUL
                        threshold_cycle = (self.eol_threshold - intercept) / slope
                        last_cycle = df['cycle'].max()
                        rul = max(1, threshold_cycle - last_cycle)
                        
                        # FIXED: More realistic bounds based on battery state and data
                        if current_soh > 90:
                            max_rul = min(1000, data_length * 3)
                        elif current_soh > 85:
                            max_rul = min(500, data_length * 2)
                        elif current_soh > self.eol_threshold:
                            max_rul = min(200, data_length)
                        else:
                            max_rul = 50
                        
                        min_rul = 1
                        
                        if min_rul <= rul <= max_rul:
                            # FIXED: Better confidence calculation
                            slope_confidence = min(1.0, abs(slope) * 1000)  # Normalize slope
                            confidence = r_squared * slope_confidence
                            
                            print(f"    ✅ Valid RUL: {rul:.0f}, confidence: {confidence:.3f}")
                            
                            if confidence > best_confidence:
                                best_rul = int(rul)
                                best_confidence = confidence
                        else:
                            print(f"    ❌ RUL {rul:.0f} outside bounds [{min_rul}, {max_rul}]")
                                
                except Exception as e:
                    print(f"    ⚠️ Error in window {window_size}: {e}")
                    continue
        
        print(f"📈 Best trend RUL: {best_rul} (confidence: {best_confidence:.3f})")
        return best_rul

    def _estimate_rul_linear_improved(self, df):
        """FIXED: Improved linear regression with better validation"""
        if len(df) < 5:
            return None
        
        current_soh = df['soh_percent'].iloc[-1]
        
        print(f"📊 Linear analysis for SoH: {current_soh:.1f}%")
        
        if current_soh < self.eol_threshold - 5:
            return self._estimate_eol_rul(df)
        
        # FIXED: More intelligent window sizing
        data_length = len(df)
        volatility = df['soh_percent'].std()
        
        if volatility > 10:
            window_size = min(data_length, max(20, data_length // 2))
        elif volatility > 5:
            window_size = min(data_length, max(15, data_length // 3))
        else:
            window_size = min(data_length, max(10, data_length // 4))
        
        recent_data = df.tail(window_size)
        X = recent_data['cycle'].values.reshape(-1, 1)
        y = recent_data['soh_percent'].values
        
        try:
            model = LinearRegression()
            model.fit(X, y)
            
            # Better validation
            y_pred = model.predict(X)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            slope = model.coef_[0]
            
            print(f"  📊 Linear: slope={slope:.6f}, R²={r_squared:.3f}")
            
            # FIXED: Better validation criteria
            if r_squared > 0.1 and slope < -0.001:  # Must have meaningful negative slope
                threshold_cycle = (self.eol_threshold - model.intercept_) / slope
                last_cycle = df['cycle'].max()
                rul = max(1, threshold_cycle - last_cycle)
                
                # Dynamic bounds
                if current_soh > 90:
                    max_rul = min(800, data_length * 3)
                elif current_soh > 85:
                    max_rul = min(400, data_length * 2)
                else:
                    max_rul = min(200, data_length)
                
                if 1 <= rul <= max_rul:
                    print(f"  ✅ Linear RUL: {rul:.0f}")
                    return int(rul)
                else:
                    print(f"  ❌ Linear RUL {rul:.0f} outside bounds [1, {max_rul}]")
            else:
                print(f"  ❌ Linear model failed validation")
        except Exception as e:
            print(f"  ⚠️ Linear regression error: {e}")
        
        return None

    def _estimate_rul_exponential_smoothing_improved(self, df):
        """FIXED: Improved exponential smoothing"""
        if len(df) < 10:
            return None
        
        current_soh = df['soh_percent'].iloc[-1]
        
        print(f"📉 Exponential smoothing for SoH: {current_soh:.1f}%")
        
        if current_soh < self.eol_threshold - 5:
            return self._estimate_eol_rul(df)
        
        return self._simple_exponential_smoothing_improved(df)

    def _simple_exponential_smoothing_improved(self, df):
        """FIXED: Improved manual exponential smoothing"""
        if len(df) < 10:
            return None
        
        current_soh = df['soh_percent'].iloc[-1]
        soh_values = df['soh_percent'].values
        
        # Optimize alpha parameter
        best_alpha = 0.3
        best_error = np.inf
        
        for alpha in [0.1, 0.2, 0.3, 0.4, 0.5]:
            try:
                smoothed = [soh_values[0]]
                for i in range(1, len(soh_values)):
                    smoothed.append(alpha * soh_values[i] + (1 - alpha) * smoothed[i-1])
                
                if len(smoothed) > 1:
                    error = mean_squared_error(soh_values[1:], smoothed[1:])
                    if error < best_error:
                        best_error = error
                        best_alpha = alpha
            except:
                continue
        
        # Apply best smoothing
        smoothed = [soh_values[0]]
        for i in range(1, len(soh_values)):
            smoothed.append(best_alpha * soh_values[i] + (1 - best_alpha) * smoothed[i-1])
        
        # Trend analysis on smoothed data
        if len(smoothed) >= 10:
            recent_window = min(30, len(smoothed))
            recent_smoothed = smoothed[-recent_window:]
            recent_indices = list(range(len(smoothed) - recent_window, len(smoothed)))
            
            try:
                slope, intercept, r_value, _, _ = linregress(recent_indices, recent_smoothed)
                
                print(f"  📊 Smoothed trend: slope={slope:.6f}, R²={r_value**2:.3f}")
                
                if abs(r_value) > 0.2 and slope < -0.01:  # More lenient for smoothed data
                    last_index = len(smoothed) - 1
                    threshold_index = (self.eol_threshold - intercept) / slope
                    rul = max(1, threshold_index - last_index)
                    
                    if 1 <= rul <= 500:
                        print(f"  ✅ Smoothed RUL: {rul:.0f}")
                        return int(rul)
                    else:
                        print(f"  ❌ Smoothed RUL {rul:.0f} outside bounds [1, 500]")
            except Exception as e:
                print(f"  ⚠️ Smoothing error: {e}")
        
        return None

    def _estimate_rul_polynomial_improved(self, df, degree=2):
        """FIXED: Improved polynomial regression"""
        if len(df) < degree + 10:
            return None
        
        current_soh = df['soh_percent'].iloc[-1]
        
        print(f"🔢 Polynomial analysis for SoH: {current_soh:.1f}%")
        
        if current_soh < self.eol_threshold - 5:
            return self._estimate_eol_rul(df)
        
        try:
            # Use more recent data
            recent_data = df.tail(min(60, len(df)))
            X = recent_data['cycle'].values.reshape(-1, 1)
            y = recent_data['soh_percent'].values
            
            poly_model = make_pipeline(
                PolynomialFeatures(degree=degree, include_bias=False),
                Ridge(alpha=1.0)
            )
            
            poly_model.fit(X, y)
            y_pred = poly_model.predict(X)
            
            # Better model validation
            r2_score = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
            
            print(f"  📊 Polynomial R²: {r2_score:.3f}")
            
            if r2_score > 0.05:  # More lenient threshold
                # Conservative future prediction
                max_cycle = df['cycle'].max()
                future_range = min(300, len(df) * 2)
                future_cycles = np.arange(max_cycle + 1, max_cycle + future_range + 1).reshape(-1, 1)
                future_soh = poly_model.predict(future_cycles)
                
                # Better clipping
                future_soh = np.clip(future_soh, 0, 110)
                
                # Find EOL
                below_threshold = np.where(future_soh <= self.eol_threshold)[0]
                if len(below_threshold) > 0:
                    rul = below_threshold[0] + 1
                    max_rul = 600 if current_soh > 90 else 300 if current_soh > 85 else 150
                    
                    if 1 <= rul <= max_rul:
                        print(f"  ✅ Polynomial RUL: {rul}")
                        return int(rul)
                    else:
                        print(f"  ❌ Polynomial RUL {rul} outside bounds [1, {max_rul}]")
            else:
                print(f"  ❌ Polynomial model failed validation")
            
        except Exception as e:
            print(f"  ⚠️ Polynomial error: {e}")
        
        return None

    def _estimate_rul_advanced_trend(self, df):
        """FIXED: Advanced trend analysis"""
        if len(df) < 15:
            return None
        
        current_soh = df['soh_percent'].iloc[-1]
        
        print(f"🚀 Advanced trend analysis for SoH: {current_soh:.1f}%")
        
        if current_soh < self.eol_threshold - 5:
            return self._estimate_eol_rul(df)
        
        # Approach 1: Multi-segment linear regression
        try:
            # Divide data into segments and analyze each
            data_length = len(df)
            segment_size = max(20, data_length // 3)
            
            best_rul = None
            best_score = 0
            
            for start_idx in range(0, data_length - segment_size + 1, segment_size // 2):
                end_idx = min(start_idx + segment_size, data_length)
                segment_data = df.iloc[start_idx:end_idx]
                
                if len(segment_data) >= 10:
                    cycles = segment_data['cycle'].values
                    soh_values = segment_data['soh_percent'].values
                    
                    slope, intercept, r_value, _, _ = linregress(cycles, soh_values)
                    r_squared = r_value ** 2
                    
                    if r_squared > 0.1 and slope < -0.001:
                        threshold_cycle = (self.eol_threshold - intercept) / slope
                        last_cycle = df['cycle'].max()
                        rul = max(1, threshold_cycle - last_cycle)
                        
                        # Score based on R² and recency of data
                        recency_weight = (start_idx + end_idx) / (2 * data_length)  # Higher for recent data
                        score = r_squared * recency_weight
                        
                        max_rul = 400 if current_soh > 85 else 200
                        if 1 <= rul <= max_rul and score > best_score:
                            best_rul = int(rul)
                            best_score = score
            
            if best_rul:
                print(f"  ✅ Multi-segment RUL: {best_rul} (score: {best_score:.3f})")
                return best_rul
                
        except Exception as e:
            print(f"  ⚠️ Multi-segment error: {e}")
        
        # Approach 2: Weighted recent trend
        try:
            # Give more weight to recent data points
            recent_data = df.tail(min(40, len(df)))
            cycles = recent_data['cycle'].values
            soh_values = recent_data['soh_percent'].values
            
            # Create weights (more recent = higher weight)
            weights = np.linspace(0.5, 1.0, len(cycles))
            
            # Weighted linear regression
            from sklearn.linear_model import LinearRegression
            
            model = LinearRegression()
            model.fit(cycles.reshape(-1, 1), soh_values, sample_weight=weights)
            
            slope = model.coef_[0]
            intercept = model.intercept_
            
            # Calculate weighted R²
            y_pred = model.predict(cycles.reshape(-1, 1))
            weighted_ss_res = np.sum(weights * (soh_values - y_pred) ** 2)
            weighted_ss_tot = np.sum(weights * (soh_values - np.average(soh_values, weights=weights)) ** 2)
            weighted_r2 = 1 - (weighted_ss_res / weighted_ss_tot) if weighted_ss_tot != 0 else 0
            
            print(f"  📊 Weighted trend: slope={slope:.6f}, R²={weighted_r2:.3f}")
            
            if weighted_r2 > 0.15 and slope < -0.001:
                threshold_cycle = (self.eol_threshold - intercept) / slope
                last_cycle = df['cycle'].max()
                rul = max(1, threshold_cycle - last_cycle)
                
                max_rul = 500 if current_soh > 85 else 250
                if 1 <= rul <= max_rul:
                    print(f"  ✅ Weighted RUL: {rul}")
                    return int(rul)
                else:
                    print(f"  ❌ Weighted RUL {rul} outside bounds [1, {max_rul}]")
            
        except Exception as e:
            print(f"  ⚠️ Weighted trend error: {e}")
        
        return None

    def _adaptive_ensemble_improved(self, valid_predictions, model_reliability, df):
        """FIXED: Better ensemble method"""
        if len(valid_predictions) == 1:
            return list(valid_predictions.values())[0]
        
        current_soh = df['soh_percent'].iloc[-1]
        
        print(f"🎯 Ensemble from {len(valid_predictions)} models: {valid_predictions}")
        
        # Filter unrealistic predictions
        realistic_predictions = {}
        
        for model, prediction in valid_predictions.items():
            # FIXED: More reasonable bounds based on current state
            if current_soh > 90:
                min_realistic_rul = max(10, (current_soh - self.eol_threshold) * 3)
                max_realistic_rul = min(1000, len(df) * 5)
            elif current_soh > 85:
                min_realistic_rul = max(5, (current_soh - self.eol_threshold) * 2)
                max_realistic_rul = min(500, len(df) * 3)
            elif current_soh > self.eol_threshold:
                min_realistic_rul = max(3, (current_soh - self.eol_threshold) * 1.5)
                max_realistic_rul = min(300, len(df) * 2)
            else:
                min_realistic_rul = 1
                max_realistic_rul = 50
            
            if min_realistic_rul <= prediction <= max_realistic_rul:
                realistic_predictions[model] = prediction
            else:
                print(f"  ❌ Filtered out {model}: {prediction} (bounds: [{min_realistic_rul:.0f}, {max_realistic_rul:.0f}])")
        
        if not realistic_predictions:
            print("  ⚠️ No realistic predictions, using median of all")
            return int(np.median(list(valid_predictions.values())))
        
        print(f"  ✅ Realistic predictions: {realistic_predictions}")
        
        # Weighted ensemble
        total_weight = 0
        weighted_sum = 0
        
        for model, prediction in realistic_predictions.items():
            base_weight = model_reliability.get(model, 0.5)
            weighted_sum += prediction * base_weight
            total_weight += base_weight
        
        final_prediction = int(weighted_sum / total_weight) if total_weight > 0 else int(np.median(list(realistic_predictions.values())))
        print(f"  🎯 Final ensemble: {final_prediction} cycles")
        return final_prediction

    def get_final_prediction(self, individual_predictions, model_reliability=None, method='intelligent'):
        """FIXED: Enhanced prediction selection"""
        valid_preds = {k: v for k, v in individual_predictions.items() if v is not None and v > 0}
        
        if not valid_preds:
            return None, 'None', 0
        
        if len(valid_preds) == 1:
            model_name = list(valid_preds.keys())[0]
            return valid_preds[model_name], model_name, 75
        
        # Better outlier detection using IQR
        pred_values = list(valid_preds.values())
        q1, q3 = np.percentile(pred_values, [25, 75])
        iqr = q3 - q1
        
        if iqr > 0:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            filtered_preds = {k: v for k, v in valid_preds.items() 
                            if lower_bound <= v <= upper_bound}
        else:
            filtered_preds = valid_preds
        
        if not filtered_preds:
            filtered_preds = valid_preds
        
        # Enhanced confidence calculation
        pred_values = list(filtered_preds.values())
        pred_std = np.std(pred_values)
        pred_mean = np.mean(pred_values)
        
        # Coefficient of variation
        cv = pred_std / pred_mean if pred_mean > 0 else 1
        
        # Base confidence on agreement
        if cv < 0.05:
            confidence = 95
        elif cv < 0.1:
            confidence = 90
        elif cv < 0.2:
            confidence = 80
        elif cv < 0.3:
            confidence = 70
        else:
            confidence = 60
        
        # Adjust for number of models
        model_count_bonus = min(15, len(filtered_preds) * 3)
        confidence = min(98, confidence + model_count_bonus)
        
        # Select best model
        if model_reliability:
            best_model = max(filtered_preds.keys(), 
                           key=lambda x: model_reliability.get(x, 0.5))
            return filtered_preds[best_model], best_model, confidence
        
        # Default priority
        model_priority = ['advanced_trend', 'trend_analysis', 'linear', 'exponential_smoothing', 'polynomial']
        for model_name in model_priority:
            if model_name in filtered_preds:
                return filtered_preds[model_name], model_name, confidence
        
        return None, 'None', 0

    def backtest_comprehensive(self, df, test_points=None, enable_progress=True):
        """FIXED: Enhanced backtesting"""
        if test_points is None:
            max_cycle = df['cycle'].max()
            
            # Better test point selection
            start_point = max(30, int(max_cycle * 0.3))  # Start from 30% of data
            end_point = int(max_cycle * 0.85)  # End at 85% to have meaningful RUL
            
            num_test_points = min(12, max(3, (end_point - start_point) // 10))
            if num_test_points > 2:
                test_points = np.linspace(start_point, end_point, num_test_points, dtype=int).tolist()
            else:
                test_points = [start_point, end_point]
        
        # Find actual EOL
        actual_eol_df = df[df['soh_percent'] <= self.eol_threshold]
        if actual_eol_df.empty:
            if enable_progress:
                print(f"⚠️ Warning: Battery never reached EOL threshold in dataset.")
                print(f"   Minimum SoH reached: {df['soh_percent'].min():.1f}%")
            actual_eol_cycle = self._extrapolate_eol_cycle(df)
        else:
            actual_eol_cycle = actual_eol_df['cycle'].iloc[0]
        
        results = []
        
        if enable_progress:
            print(f"\n🔬 Starting Enhanced RUL Backtesting...")
            print(f"📊 Battery Type: {self.battery_type}")
            print(f"🎯 EOL Threshold: {self.eol_threshold}%")
            print(f"📈 Actual/Estimated EOL at cycle: {actual_eol_cycle}")
            print(f"🧪 Testing at {len(test_points)} points")
            print("-" * 70)
        
        for i, test_cycle in enumerate(test_points):
            if test_cycle >= actual_eol_cycle:
                continue
            
            if enable_progress:
                progress = (i + 1) / len(test_points) * 100
                print(f"\n📍 Progress: {progress:.1f}% - Testing cycle {test_cycle}")
            
            historical_df = df[df['cycle'] <= test_cycle]
            actual_rul = actual_eol_cycle - test_cycle
            
            try:
                ensemble_rul, individual_predictions, model_reliability = self.estimate_rul_ensemble(historical_df)
                final_rul_prediction, final_method_used, confidence = self.get_final_prediction(
                    individual_predictions, model_reliability, method='intelligent'
                )
            except Exception as e:
                if enable_progress:
                    print(f"   ⚠️ Error at cycle {test_cycle}: {str(e)}")
                continue
            
            # Calculate metrics
            health_score = self.calculate_health_score(historical_df)
            health_status = self.get_health_status(historical_df['soh_percent'].iloc[-1])
            
            # Anomaly detection
            anomalies_dict = self.detect_anomalies(historical_df)
            anomaly_count = anomalies_dict.get('count', 0)
            
            # Store results
            result = {
                'test_cycle': test_cycle,
                'actual_rul': actual_rul,
                'ensemble_rul': ensemble_rul,
                'ensemble_error': (ensemble_rul - actual_rul) if ensemble_rul else None,
                'final_rul': final_rul_prediction,
                'final_method': final_method_used,
                'final_error': (final_rul_prediction - actual_rul) if final_rul_prediction else None,
                'prediction_confidence': confidence,
                'health_score': health_score,
                'health_status': health_status,
                'anomaly_count': anomaly_count,
                'data_points_used': len(historical_df)
            }
            
            # Add individual model results
            for method, pred in individual_predictions.items():
                result[f'{method}_rul'] = pred
                result[f'{method}_error'] = (pred - actual_rul) if pred else None
            
            results.append(result)
            
            if enable_progress:
                print(f"   🎯 Final Prediction ({final_method_used}): {final_rul_prediction} cycles")
                print(f"   📊 Actual RUL: {actual_rul} cycles (Error: {result['final_error']})")
                print(f"   💊 Health: {health_score:.1f} | Status: {health_status} | Confidence: {confidence}%")
        
        results_df = pd.DataFrame(results)
        
        if enable_progress and not results_df.empty:
            print("\n" + "="*70)
            print("📊 ENHANCED BACKTESTING SUMMARY")
            print("="*70)
            
            # Performance metrics
            final_errors = results_df['final_error'].dropna()
            if not final_errors.empty:
                mae = final_errors.abs().mean()
                rmse = np.sqrt((final_errors ** 2).mean())
                mape = (final_errors.abs() / results_df['actual_rul']).mean() * 100
                
                print(f"🎯 Final Model Performance:")
                print(f"   • Mean Absolute Error (MAE): {mae:.2f} cycles")
                print(f"   • Root Mean Square Error (RMSE): {rmse:.2f} cycles")
                print(f"   • Mean Absolute Percentage Error (MAPE): {mape:.1f}%")
                print(f"   • Average Confidence: {results_df['prediction_confidence'].mean():.1f}%")
        
        return results_df

    def _extrapolate_eol_cycle(self, df):
        """FIXED: Better EOL extrapolation"""
        try:
            # Use recent trend for extrapolation
            recent_data = df.tail(min(50, len(df)))
            if len(recent_data) >= 10:
                X = recent_data['cycle'].values.reshape(-1, 1)
                y = recent_data['soh_percent'].values
                
                model = LinearRegression()
                model.fit(X, y)
                
                if model.coef_[0] < -0.001:  # Valid degradation
                    eol_cycle = (self.eol_threshold - model.intercept_) / model.coef_[0]
                    return max(df['cycle'].max() + 5, eol_cycle)
            
            # Fallback: use average degradation rate
            current_soh = df['soh_percent'].iloc[-1]
            if len(df) >= 20:
                total_degradation = df['soh_percent'].iloc[0] - current_soh
                degradation_rate = total_degradation / len(df)
                if degradation_rate > 0:
                    remaining_degradation = current_soh - self.eol_threshold
                    additional_cycles = remaining_degradation / degradation_rate
                    return df['cycle'].max() + max(5, additional_cycles)
            
            # Conservative estimate
            remaining_capacity = current_soh - self.eol_threshold
            return df['cycle'].max() + max(10, remaining_capacity * 2)
            
        except Exception:
            return df['cycle'].max() + 20

    def plot_rul_predictions(self, df, test_cycle, save_plot=False, return_fig=False):
        """Enhanced visualization with better scaling and information"""
        historical_df = df[df['cycle'] <= test_cycle]
        ensemble_rul, individual_predictions, model_reliability = self.estimate_rul_ensemble(historical_df)
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # Main SoH plot
        ax1.plot(df['cycle'], df['soh_percent'], 'b-', label='Actual SoH', linewidth=2.5, alpha=0.8)
        ax1.axvline(x=test_cycle, color='orange', linestyle='--', alpha=0.8, 
                   label=f'Test Point (Cycle {test_cycle})', linewidth=2)
        ax1.axhline(y=self.eol_threshold, color='red', linestyle=':', alpha=0.8, 
                   label=f'EOL Threshold ({self.eol_threshold}%)', linewidth=2)
        
        # Individual model predictions
        colors = ['green', 'purple', 'brown', 'pink', 'gray', 'cyan', 'olive']
        for i, (method, rul) in enumerate(individual_predictions.items()):
            if rul and rul > 0:
                predicted_eol = test_cycle + rul
                reliability = model_reliability.get(method, 0.5)
                alpha = 0.3 + 0.6 * reliability
                ax1.axvline(x=predicted_eol, color=colors[i % len(colors)], 
                           linestyle='-.', alpha=alpha, linewidth=2,
                           label=f'{method.replace("_", " ").title()}: {rul} cycles (R={reliability:.2f})')
        
        # Ensemble prediction
        if ensemble_rul and ensemble_rul > 0:
            ensemble_eol = test_cycle + ensemble_rul
            ax1.axvline(x=ensemble_eol, color='red', linestyle='-', linewidth=4, 
                       alpha=0.9, label=f'Ensemble: {ensemble_rul} cycles')
        
        ax1.set_xlabel('Cycle Number', fontsize=12)
        ax1.set_ylabel('State of Health (%)', fontsize=12)
        ax1.set_title(f'RUL Predictions at Cycle {test_cycle} - {self.battery_type}', fontsize=14, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(max(0, self.eol_threshold - 10), 105)
        
        # Health score evolution
        health_scores = []
        cycles_for_health = []
        step_size = max(1, len(historical_df) // 30)
        
        for i in range(15, len(historical_df), step_size):
            subset_df = historical_df.iloc[:i+1]
            health_score = self.calculate_health_score(subset_df)
            health_scores.append(health_score)
            cycles_for_health.append(subset_df['cycle'].iloc[-1])
        
        if health_scores:
            ax2.plot(cycles_for_health, health_scores, 'g-', linewidth=2, marker='o', markersize=3)
            ax2.axhline(y=85, color='orange', linestyle='--', alpha=0.7, label='Good Health (85)')
            ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Fair Health (70)')
            ax2.axhline(y=50, color='darkred', linestyle='--', alpha=0.7, label='Critical Health (50)')
            ax2.set_xlabel('Cycle Number', fontsize=12)
            ax2.set_ylabel('Health Score', fontsize=12)
            ax2.set_title('Battery Health Score Evolution', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 100)
        
        # Degradation rate analysis
        if len(df) >= 20:
            window_size = min(20, len(df) // 5)
            degradation_rates = []
            cycles_for_deg = []
            
            for i in range(window_size, len(df)):
                window_data = df.iloc[i-window_size:i]
                if len(window_data) > 1:
                    slope, _, _, _, _ = linregress(window_data['cycle'], window_data['soh_percent'])
                    degradation_rates.append(-slope)  # Make positive for degradation
                    cycles_for_deg.append(window_data['cycle'].iloc[-1])
            
            if degradation_rates:
                ax3.plot(cycles_for_deg, degradation_rates, 'r-', linewidth=2, marker='s', markersize=2)
                ax3.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Normal Degradation (0.1%/cycle)')
                ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='High Degradation (0.5%/cycle)')
                ax3.set_xlabel('Cycle Number', fontsize=12)
                ax3.set_ylabel('Degradation Rate (%/cycle)', fontsize=12)
                ax3.set_title('Degradation Rate Analysis', fontsize=12, fontweight='bold')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                ax3.set_ylim(0, max(1.0, max(degradation_rates) * 1.1) if degradation_rates else 1.0)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(f'enhanced_rul_prediction_cycle_{test_cycle}.png', dpi=300, bbox_inches='tight')
        
        if return_fig:
            return fig
        
        plt.show()

    def create_plotly_dashboard(self, df):
        """Enhanced interactive Plotly dashboard"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('State of Health Degradation', 'Capacity & Voltage Trends',
                           'Health Score Evolution', 'Temperature Analysis',
                           'Degradation Rate', 'Anomaly Detection'),
            specs=[[{"secondary_y": False}, {"secondary_y": True}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # SoH degradation plot
        fig.add_trace(
            go.Scatter(x=df['cycle'], y=df['soh_percent'], 
                      mode='lines+markers', name='SoH (%)',
                      line=dict(color='blue', width=3),
                      marker=dict(size=4)),
            row=1, col=1
        )
        
        fig.add_hline(y=self.eol_threshold, line_dash="dash", line_color="red",
                     annotation_text=f"EOL Threshold ({self.eol_threshold}%)",
                     row=1, col=1)
        
        # Add other traces for available columns
        if 'capacity_ah' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['cycle'], y=df['capacity_ah'], 
                          mode='lines', name='Capacity (Ah)',
                          line=dict(color='green', width=2)),
                row=1, col=2
            )
        
        if 'avg_voltage' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['cycle'], y=df['avg_voltage'], 
                          mode='lines', name='Voltage (V)',
                          line=dict(color='purple', width=2)),
                row=1, col=2, secondary_y=True
            )
        
        # Health score evolution
        health_scores = []
        for i in range(15, len(df), max(1, len(df)//50)):
            subset_df = df.iloc[:i+1]
            health_score = self.calculate_health_score(subset_df)
            health_scores.append(health_score)
        
        health_cycles = df['cycle'].iloc[14::max(1, len(df)//50)][:len(health_scores)]
        
        fig.add_trace(
            go.Scatter(x=health_cycles, y=health_scores,
                      mode='lines+markers', name='Health Score',
                      line=dict(color='orange', width=2),
                      marker=dict(size=4)),
            row=2, col=1
        )
        
        # Temperature analysis (if available)
        if 'avg_temp_c' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['cycle'], y=df['avg_temp_c'],
                          mode='lines', name='Temperature (°C)',
                          line=dict(color='red', width=2)),
                row=2, col=2
            )
        
        # Degradation rate
        if len(df) >= 20:
            window_size = min(15, len(df) // 5)
            degradation_rates = []
            cycles_for_deg = []
            
            for i in range(window_size, len(df)):
                window_data = df.iloc[i-window_size:i]
                if len(window_data) > 1:
                    slope, _, _, _, _ = linregress(window_data['cycle'], window_data['soh_percent'])
                    degradation_rates.append(-slope * 100)  # Convert to %/cycle
                    cycles_for_deg.append(window_data['cycle'].iloc[-1])
            
            if degradation_rates:
                fig.add_trace(
                    go.Scatter(x=cycles_for_deg, y=degradation_rates,
                              mode='lines+markers', name='Degradation Rate',
                              line=dict(color='darkred', width=2),
                              marker=dict(size=3)),
                    row=3, col=1
                )
        
        # Anomaly detection
        anomalies_dict = self.detect_anomalies(df)
        anomalies_cycles = anomalies_dict.get('cycles', [])
        
        if anomalies_cycles:
            anomaly_soh = []
            for cycle in anomalies_cycles:
                matching_rows = df[df['cycle'] == cycle]
                if not matching_rows.empty:
                    anomaly_soh.append(matching_rows['soh_percent'].iloc[0])
                else:
                    anomaly_soh.append(df['soh_percent'].mean())
            
            fig.add_trace(
                go.Scatter(x=anomalies_cycles, y=anomaly_soh,
                          mode='markers', name='Anomalies',
                          marker=dict(color='red', size=8, symbol='x')),
                row=3, col=2
            )
        
        # Add normal operation line
        fig.add_trace(
            go.Scatter(x=df['cycle'], y=df['soh_percent'],
                      mode='lines', name='Normal Operation',
                      line=dict(color='blue', width=1, dash='dot'),
                      showlegend=False),
            row=3, col=2
        )
        
        fig.update_layout(height=1000, showlegend=True,
                         title_text=f"Enhanced Battery Health Dashboard - {self.battery_type}")
        
        return fig


def run_enhanced_analysis(mat_file_path, eol_threshold=80, battery_type="Li-ion 18650"):
    """
    Corrected and complete analysis pipeline.
    """
    print(f"\n" + "="*70)
    print(f"🔋 SMART BATTERY HEALTH TRACKER - ENHANCED ANALYSIS")
    print(f"📁 Processing: {getattr(mat_file_path, 'name', str(mat_file_path))}")
    print(f"🔋 Battery Type: {battery_type}")
    print(f"🎯 EOL Threshold: {eol_threshold}%")
    print("="*70)
    
    try:
        # --- 1. Load Data ---
        print("🔄 Processing discharge cycles...")
        if hasattr(mat_file_path, 'read'):
            import io
            mat = scipy.io.loadmat(io.BytesIO(mat_file_path.getvalue()))
        else:
            mat = scipy.io.loadmat(mat_file_path)

        battery_key = [k for k in mat.keys() if not k.startswith('__')][0]
        battery = mat[battery_key][0, 0]
        cycles = battery['cycle'][0]
        
        # --- 2. Process Cycles into a Clean List ---
        processed_data = []
        initial_capacity = None
        cycle_counter = 0

        for cycle_struct in cycles:
            if get_cycle_type(cycle_struct) == 'd': # Use the safe helper function
                cycle_counter += 1
                data_dict = cycle_struct['data'][0, 0]
                
                # Define all data variables correctly
                currents = data_dict['Current_measured'].flatten()
                voltages = data_dict['Voltage_measured'].flatten()
                temps = data_dict['Temperature_measured'].flatten()
                times = data_dict['Time'].flatten()
                
                capacity = np.trapezoid(currents, x=times) / 3600
                
                if initial_capacity is None:
                    initial_capacity = abs(capacity)
                
                soh = (abs(capacity) / initial_capacity) * 100
                
                processed_data.append({
                    'cycle': cycle_counter,
                    'capacity_ah': abs(capacity),
                    'soh_percent': soh,
                    'avg_voltage': np.mean(voltages),
                    'avg_current': np.mean(currents),
                    'avg_temp_c': np.mean(temps)
                    # Add any other fields you need here
                })

        if not processed_data:
            raise ValueError("No valid discharge cycle data was found in the .mat file.")

        # --- 3. Create DataFrame and Run Analysis ---
        df = pd.DataFrame(processed_data)
        print(f"✅ Processed {len(df)} discharge cycles")
        
        predictor = AdvancedBatteryRULPredictor(eol_threshold=eol_threshold, battery_type=battery_type)
        
        print("\n🧠 Running ML-based health assessment...")
        
        # Calculate health metrics
        health_score = predictor.calculate_health_score(df)
        current_soh = df['soh_percent'].iloc[-1]
        health_status = predictor.get_health_status(current_soh)
        
        print(f"🚨 Running anomaly detection...")
        
        # Anomaly detection
        anomalies_dict = predictor.detect_anomalies(df)
        anomalies_df = pd.DataFrame(anomalies_dict['anomalies']) if anomalies_dict['anomalies'] else pd.DataFrame()
        
        print(f"🔬 Performing backtesting analysis...")
        
        # Backtesting
        results_df = predictor.backtest_comprehensive(df, enable_progress=False)
        
        # Final RUL prediction using all available data
        ensemble_rul, individual_predictions, model_reliability = predictor.estimate_rul_ensemble(df)
        final_rul_prediction, final_method_used, confidence = predictor.get_final_prediction(
            individual_predictions, model_reliability, method='intelligent'
        )
        
        # Analysis summary
        capacity_fade = 100 - current_soh
        analysis_summary = {
            'total_cycles': len(df),
            'current_soh': current_soh,
            'initial_capacity': initial_capacity,
            'capacity_fade': capacity_fade,
            'anomaly_count': len(anomalies_df),
            'health_score': health_score,
            'final_rul': final_rul_prediction,
            'confidence': confidence
        }
        
        print("="*70)
        print("✅ ANALYSIS COMPLETE - SUMMARY (FIXED)")
        print("="*70)
        print(f"🎯 Final RUL Prediction: {final_rul_prediction} cycles ({final_method_used})")
        print(f"🏥 Health Score: {health_score:.1f}/100 ({health_status})")
        print(f"📊 Prediction Confidence: {confidence}%")
        print(f"🚨 Anomalies Detected: {len(anomalies_df)}")
        print(f"📈 Current SoH: {current_soh:.1f}%")
        print(f"⚡ Capacity Fade: {capacity_fade:.1f}%")
        print(f"🔧 Model Performance:")
        print(f"   • Individual Predictions: {individual_predictions}")
        print(f"   • Model Reliabilities: {model_reliability}")
        print(f"   • Selected Method: {final_method_used}")
        print("="*70)
        
        return {
            "dataframe": df,
            "predictor": predictor,
            "backtest_results": results_df,
            "final_prediction": final_rul_prediction,
            "final_method": final_method_used,
            "individual_predictions": individual_predictions,
            "model_reliability": model_reliability,
            "health_score": health_score,
            "health_status": health_status,
            "anomalies": anomalies_df,
            "prediction_confidence": confidence,
            "analysis_summary": analysis_summary
        }
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# Example usage and testing
if __name__ == "__main__":
    # Test the enhanced predictor
    try:
        # Create sample data for testing
        np.random.seed(42)
        cycles = np.arange(1, 201)
        
        # Simulate realistic battery degradation
        initial_capacity = 2.0
        degradation_rate = 0.001  # 0.1% per cycle
        noise_level = 0.02
        
        capacities = []
        for cycle in cycles:
            # Non-linear degradation with some noise
            base_degradation = initial_capacity * (1 - degradation_rate * cycle - 0.000001 * cycle**2)
            noise = np.random.normal(0, noise_level)
            capacity = max(0.5, base_degradation + noise)
            capacities.append(capacity)
        
        # Create test DataFrame
        test_df = pd.DataFrame({
            'cycle': cycles,
            'capacity_ah': capacities,
            'avg_voltage': 3.7 + np.random.normal(0, 0.1, len(cycles)),
            'avg_temp_c': 25 + np.random.normal(0, 5, len(cycles)),
            'soh_percent': (np.array(capacities) / initial_capacity) * 100
        })
        
        print("🧪 Testing Enhanced Battery RUL Predictor...")
        print(f"📊 Test data: {len(test_df)} cycles, SoH range: {test_df['soh_percent'].min():.1f}% - {test_df['soh_percent'].max():.1f}%")
        
        # Initialize predictor
        predictor = AdvancedBatteryRULPredictor(eol_threshold=80)
        
        # Test RUL prediction
        ensemble_rul, individual_preds, reliabilities = predictor.estimate_rul_ensemble(test_df)
        final_rul, method, confidence = predictor.get_final_prediction(individual_preds, reliabilities)
        
        print(f"\n✅ Test Results:")
        print(f"🎯 Final RUL: {final_rul} cycles ({method})")
        print(f"📊 Individual predictions: {individual_preds}")
        print(f"🏥 Health score: {predictor.calculate_health_score(test_df):.1f}")
        print(f"📈 Confidence: {confidence}%")
        
        # Test backtesting
        print(f"\n🔬 Running backtesting...")
        backtest_results = predictor.backtest_comprehensive(test_df, enable_progress=True)
        
        if not backtest_results.empty:
            print(f"📋 Backtesting completed with {len(backtest_results)} test points")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()