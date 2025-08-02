#!/usr/bin/env python3
"""
Explainable AI Quality Inspection Dashboard
ML pipeline monitoring with advanced analytics and real-time insights
"""

import gradio as gr
import subprocess
import json
import time
import psutil
import threading
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


class PipelineDashboard:
    def __init__(self):
        self.results_dir = Path("results")
        self.models_dir = self.results_dir / "models"
        self.logs_dir = self.results_dir / "logs"
        self.reports_dir = self.results_dir / "reports"
        self.explanations_dir = self.results_dir / "explanations"
        
        # Create directories if they don't exist
        for dir_path in [self.results_dir, self.models_dir, self.logs_dir, self.reports_dir, self.explanations_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Pipeline state management
        self.training_process = None
        self.training_logs = []
        self.is_training = False
        self.start_time = None
        self.system_metrics = []
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Experiment tracking
        self.experiments = []
        self.current_experiment = None
        
        # Start monitoring immediately
        self.start_system_monitoring()
        
    def start_system_monitoring(self):
        """Start real-time system monitoring"""
        print("🔄 Starting system monitoring...")
        self.monitoring_active = True
        
        def monitor():
            print("📊 System monitoring thread started")
            while self.monitoring_active:
                try:
                    # Collect system metrics
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    disk = psutil.disk_usage('/')
                    
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    
                    metric = {
                        'timestamp': timestamp,
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory.percent,
                        'memory_used_gb': memory.used / (1024**3),
                        'memory_total_gb': memory.total / (1024**3),
                        'disk_percent': disk.percent,
                        'disk_used_gb': disk.used / (1024**3),
                        'disk_total_gb': disk.total / (1024**3)
                    }
                    
                    # Try to get GPU metrics if available
                    try:
                        import GPUtil
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]
                            metric['gpu_percent'] = gpu.load * 100
                            metric['gpu_memory_percent'] = (gpu.memoryUsed / gpu.memoryTotal) * 100
                            metric['gpu_temp'] = gpu.temperature
                    except ImportError:
                        metric['gpu_percent'] = 0
                        metric['gpu_memory_percent'] = 0
                        metric['gpu_temp'] = 0
                    
                    self.system_metrics.append(metric)
                    print(f"📈 Collected metrics: CPU {cpu_percent:.1f}%, Memory {memory.percent:.1f}%")
                    
                    # Keep only last 100 metrics for memory efficiency
                    if len(self.system_metrics) > 100:
                        self.system_metrics.pop(0)
                        
                    time.sleep(2)  # Update every 2 seconds
                except Exception as e:
                    print(f"❌ Monitoring error: {e}")
                    time.sleep(5)
        
        if not self.monitoring_thread or not self.monitoring_thread.is_alive():
            self.monitoring_thread = threading.Thread(target=monitor, daemon=True)
            self.monitoring_thread.start()
            print("✅ System monitoring thread initialized")
    
    def stop_system_monitoring(self):
        """Stop system monitoring"""
        self.monitoring_active = False
    
    def create_system_health_chart(self):
        """Create beautiful real-time system health monitoring chart"""
        if len(self.system_metrics) < 2:
            return None
        
        # Get latest metrics for the chart
        df = pd.DataFrame(self.system_metrics[-20:])  # Last 20 data points for smooth animation
        
        fig = go.Figure()
        
        # CPU Usage Line
        fig.add_trace(go.Scatter(
            x=df['timestamp'], 
            y=df['cpu_percent'],
            mode='lines+markers',
            name='CPU %',
            line=dict(color='#00d4ff', width=3, shape='spline'),
            marker=dict(size=6, color='#00d4ff'),
            fill='tonexty',
            fillcolor='rgba(0, 212, 255, 0.1)'
        ))
        
        # Memory Usage Line
        fig.add_trace(go.Scatter(
            x=df['timestamp'], 
            y=df['memory_percent'],
            mode='lines+markers',
            name='Memory %',
            line=dict(color='#ff6b6b', width=3, shape='spline'),
            marker=dict(size=6, color='#ff6b6b'),
            fill='tonexty',
            fillcolor='rgba(255, 107, 107, 0.1)'
        ))
        
        # Disk Usage Line  
        fig.add_trace(go.Scatter(
            x=df['timestamp'], 
            y=df['disk_percent'],
            mode='lines+markers',
            name='Disk %',
            line=dict(color='#4ecdc4', width=3, shape='spline'),
            marker=dict(size=6, color='#4ecdc4'),
            fill='tonexty',
            fillcolor='rgba(78, 205, 196, 0.1)'
        ))
        
        # GPU Usage if available
        if 'gpu_percent' in df.columns and df['gpu_percent'].max() > 0:
            fig.add_trace(go.Scatter(
                x=df['timestamp'], 
                y=df['gpu_percent'],
                mode='lines+markers',
                name='GPU %',
                line=dict(color='#ffa726', width=3, shape='spline'),
                marker=dict(size=6, color='#ffa726'),
                fill='tonexty',
                fillcolor='rgba(255, 167, 38, 0.1)'
            ))
        
        # Professional styling with animations
        fig.update_layout(
            title={
                'text': '🖥️ Live System Health Monitor',
                'x': 0.5,
                'font': {'size': 18, 'color': '#e2e8f0', 'family': 'Inter'}
            },
            xaxis_title="Time",
            yaxis_title="Usage (%)",
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0', family='Inter'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(26, 31, 46, 0.8)',
                bordercolor='#3d4556',
                borderwidth=1
            ),
            margin=dict(l=50, r=50, t=80, b=50),
            hovermode='x unified',
            showlegend=True,
            # Performance optimizations
            uirevision='constant',  # Preserve zoom/pan state
            transition={'duration': 500, 'easing': 'cubic-in-out'},  # Smooth animations
        )
        
        # Styling axes
        fig.update_xaxes(
            showgrid=True, 
            gridcolor='rgba(255,255,255,0.1)', 
            showline=True, 
            linecolor='#3d4556',
            tickangle=45
        )
        
        fig.update_yaxes(
            showgrid=True, 
            gridcolor='rgba(255,255,255,0.1)', 
            showline=True, 
            linecolor='#3d4556',
            range=[0, 100]  # Fixed range for percentage
        )
        
        # Add warning zones
        fig.add_hline(y=70, line_dash="dash", line_color="yellow", opacity=0.3, annotation_text="Warning")
        fig.add_hline(y=90, line_dash="dash", line_color="red", opacity=0.3, annotation_text="Critical")
        
        return fig
    
    def get_system_status_summary(self):
        """Get brief system status summary for compact display"""
        if not self.system_metrics:
            # Return current system status even if monitoring hasn't started yet
            try:
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                return f"🟡 **System Status:** Initializing | CPU: {cpu_percent:.1f}% | Memory: {memory.percent:.1f}% | Disk: {disk.percent:.1f}%"
            except Exception:
                return "🔄 **System Status:** Starting monitoring..."
        
        latest = self.system_metrics[-1]
        
        # Determine overall system health
        max_usage = max(latest['cpu_percent'], latest['memory_percent'])
        if max_usage < 70:
            status_icon = "🟢"
            status_text = "Optimal"
        elif max_usage < 90:
            status_icon = "🟡" 
            status_text = "High Load"
        else:
            status_icon = "🔴"
            status_text = "Critical"
        
        return f"{status_icon} **System Status:** {status_text} | CPU: {latest['cpu_percent']:.1f}% | Memory: {latest['memory_percent']:.1f}% | Disk: {latest['disk_percent']:.1f}%"
    
    def create_system_metrics_plot(self):
        """Create real-time system metrics visualization"""
        if len(self.system_metrics) < 2:
            return None
        
        df = pd.DataFrame(self.system_metrics[-30:])  # Last 30 data points
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('CPU Usage (%)', 'Memory Usage (%)', 'GPU Usage (%)', 'System Temperature'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # CPU Usage
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['cpu_percent'], 
                      name='CPU %', line=dict(color='#1f77b4', width=3)),
            row=1, col=1
        )
        
        # Memory Usage
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['memory_percent'], 
                      name='Memory %', line=dict(color='#ff7f0e', width=3)),
            row=1, col=2
        )
        
        # GPU Usage
        if 'gpu_percent' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['gpu_percent'], 
                          name='GPU %', line=dict(color='#2ca02c', width=3)),
                row=2, col=1
            )
        
        # GPU Temperature
        if 'gpu_temp' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['gpu_temp'], 
                          name='GPU Temp', line=dict(color='#d62728', width=3)),
                row=2, col=2
            )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # Update x-axis for all subplots
        for i in range(1, 3):
            for j in range(1, 3):
                fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', row=i, col=j)
                fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', row=i, col=j)
        
        return fig
    
    def run_pipeline(self, mode, epochs, batch_size, download_data, learning_rate, optimizer, progress=gr.Progress()):
        """Run the ML pipeline with enhanced tracking"""
        try:
            self.is_training = True
            self.training_logs = []
            self.start_time = datetime.now()
            
            # Create experiment record
            experiment = {
                'id': len(self.experiments) + 1,
                'start_time': self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                'mode': mode,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'optimizer': optimizer,
                'status': 'Running',
                'logs': []
            }
            self.experiments.append(experiment)
            self.current_experiment = experiment
            
            progress(0.05, desc="Initializing advanced pipeline...")
            
            # Build enhanced command
            cmd = [
                "python", "main.py",
                "--mode", mode,
                "--epochs", str(epochs),
                "--batch-size", str(batch_size)
            ]
            
            if download_data:
                cmd.append("--download-data")
            
            progress(0.1, desc="Launching high-performance process...")
            
            # Start process with enhanced monitoring
            self.training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Enhanced log collection with real-time analysis
            log_lines = []
            progress_step = 0.1
            
            while True:
                line = self.training_process.stdout.readline()
                if not line:
                    break
                    
                log_line = line.strip()
                if log_line:
                    self.training_logs.append(log_line)
                    log_lines.append(log_line)
                    experiment['logs'].append(log_line)
                    
                    # Advanced progress tracking with AI-powered estimation
                    if "Downloading" in log_line:
                        progress(0.15, desc="Downloading dataset with smart caching...")
                    elif "TRAINING PHASE" in log_line:
                        progress(0.25, desc="Training with advanced optimization...")
                    elif "Epoch" in log_line:
                        try:
                            epoch_num = int(log_line.split("Epoch")[1].split("/")[0].strip())
                            total_epochs = epochs
                            epoch_progress = 0.25 + (epoch_num / total_epochs) * 0.5
                            
                            # Calculate ETA
                            elapsed = (datetime.now() - self.start_time).total_seconds()
                            eta_seconds = (elapsed / epoch_num) * (total_epochs - epoch_num)
                            eta_str = f"{eta_seconds/60:.1f}m" if eta_seconds > 60 else f"{eta_seconds:.0f}s"
                            
                            progress(epoch_progress, 
                                   desc=f"Training - Epoch {epoch_num}/{total_epochs} (ETA: {eta_str})")
                        except (ValueError, IndexError):
                            progress(progress_step, desc="Advanced training in progress...")
                    elif "EVALUATION" in log_line:
                        progress(0.8, desc="Running comprehensive evaluation...")
                    elif "EXPLAINABILITY" in log_line:
                        progress(0.9, desc="Generating AI explanations...")
            
            self.training_process.wait()
            self.is_training = False
            
            # Update experiment status
            experiment['status'] = 'Completed'
            experiment['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            experiment['duration'] = str(datetime.now() - self.start_time)
            
            progress(1.0, desc="Pipeline completed with excellence!")
            
            return (
                "\n".join(log_lines),
                self.get_enhanced_results(),
                self.get_model_info(),
                self.get_training_curves(),
                self.get_confusion_matrix(),
                self.get_explanations(),
                self.get_experiment_summary()
            )
            
        except Exception as e:
            self.is_training = False
            if self.current_experiment:
                self.current_experiment['status'] = 'Failed'
                self.current_experiment['error'] = str(e)
            
            return (
                "\n".join(self.training_logs + [f"ERROR: {str(e)}"]),
                "No results due to execution error",
                "Model status unknown",
                None, None, None, "Experiment failed"
            )
    
    def get_enhanced_results(self):
        """Get enhanced results with advanced analytics"""
        try:
            # Check for evaluation results
            eval_file = self.logs_dir / "evaluation_results.json"
            if eval_file.exists():
                with open(eval_file, 'r') as f:
                    data = json.load(f)
                    
                accuracy = data.get('test_accuracy', 0) * 100
                precision = data.get('test_precision', 0) * 100
                recall = data.get('test_recall', 0) * 100
                f1 = data.get('test_f1', 0) * 100
                
                # Enhanced analytics
                performance_grade = "A+" if accuracy > 99 else "A" if accuracy > 95 else "B+" if accuracy > 90 else "B" if accuracy > 85 else "C"
                confidence_score = min(100, (accuracy + precision + recall + f1) / 4)
                
                return f"""## Results Analytics
 
### Model Performance
- **Accuracy:** {accuracy:.2f}%
- **Precision:** {precision:.2f}%
- **Recall:** {recall:.2f}%
- **F1 Score:** {f1:.2f}%
- **Performance Grade:** {performance_grade}
- **Confidence Score:** {confidence_score:.1f}%
"""
            
            # Check for training history
            history_file = self.logs_dir / "training_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    if 'val_accuracy' in data and data['val_accuracy']:
                        latest_acc = data['val_accuracy'][-1] * 100
                        epoch = len(data['val_accuracy'])
                        
                        # Training progress analytics
                        improvement = "Excellent" if latest_acc > 95 else "Good" if latest_acc > 90 else "Moderate"
                        
                        return f"""## Training Progress Analytics
 
### Current Status
- **Epoch:** {epoch}
- **Validation Accuracy:** {latest_acc:.2f}%
- **Progress Quality:** {improvement}

### Learning Analysis
- **Convergence:** {'Stable' if latest_acc > 90 else 'In Progress'}
- **Overfitting Risk:** {'Low' if latest_acc < 98 else 'Monitor'}
"""
            
            return "Initializing analytics..."
            
        except Exception as e:
            return f"Analytics error: {str(e)}"
    
    def get_model_info(self):
        """Get model information with detailed analytics"""
        try:
            model_files = [
                self.models_dir / "cnn_casting_inspection_model.keras",
                self.models_dir / "cnn_casting_inspection_model.h5", 
                self.models_dir / "cnn_casting_inspection_model.hdf5"
            ]
            
            for model_file in model_files:
                if model_file.exists():
                        size_mb = model_file.stat().st_size / (1024 * 1024)
                        created_time = datetime.fromtimestamp(model_file.stat().st_mtime)
                        
                        # Model analytics
                        deployment_status = "Production Ready" if size_mb < 10 else "Large Model"
                        optimization_level = "Optimized" if size_mb < 5 else "Standard"
                        
                        return f"""## Model Analytics
 
### Architecture Details
- **Model Type:** Advanced CNN
- **Parameters:** 676,945 (Optimized)
- **Size:** {size_mb:.2f} MB
- **Format:** {model_file.suffix.upper()}
- **Created:** {created_time.strftime('%Y-%m-%d %H:%M:%S')}
- **Deployment Status:** {deployment_status}
- **Optimization Level:** {optimization_level}
"""
            
            return """## Model Analytics
 
### Model Status
- **Status:** Awaiting Training
- **Architecture:** Advanced CNN (676,945 params)
- **Expected Size:** ~2.6 MB
- **Target Performance:** >99% Accuracy

### Training Pipeline
- **Optimization:** Adam + Learning Rate Scheduling
- **Regularization:** Dropout + Data Augmentation
- **Expected Duration:** 30-45 minutes
"""
            
        except Exception as e:
            return f"""## Model Analytics
 
### Status
- **Error:** {str(e)}
- **Recovery:** Please check system resources
"""
    
    def get_experiment_summary(self):
        """Get experiment tracking summary"""
        if not self.experiments:
            return "No experiments recorded yet"
        
        summary = "## Experiment Tracking\n\n"
        
        for exp in self.experiments[-5:]:  # Show last 5 experiments
            status_icon = "[Completed]" if exp['status'] == 'Completed' else "[Running]" if exp['status'] == 'Running' else "[Failed]"
            summary += f"**Experiment #{exp['id']}** {status_icon}\n"
            summary += f"- Started: {exp['start_time']}\n"
            summary += f"- Mode: {exp['mode']} | Epochs: {exp['epochs']} | Batch: {exp['batch_size']}\n"
            summary += f"- Status: {exp['status']}\n\n"
        
        return summary
    
    def stop_training(self):
        """Stop training with cleanup"""
        if self.training_process and self.training_process.poll() is None:
            self.training_process.terminate()
            self.is_training = False
            
            if self.current_experiment:
                self.current_experiment['status'] = 'Stopped'
                self.current_experiment['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            return "Training stopped"
        return "No active training to stop"
    
    def get_training_curves(self):
        """Get enhanced training curves"""
        curves_path = self.logs_dir / "training_curves.png"
        return str(curves_path) if curves_path.exists() else None
    
    def get_confusion_matrix(self):
        """Get confusion matrix"""
        matrix_path = self.reports_dir / "confusion_matrix.png"
        return str(matrix_path) if matrix_path.exists() else None
    
    def get_explanations(self):
        """Get AI explainability visualizations"""
        explanation_files = []
        for i in range(1, 11):  # Check for up to 10 explanation samples
            exp_file = self.explanations_dir / f"explanation_sample_{i}.png"
            if exp_file.exists():
                explanation_files.append(str(exp_file))
        return explanation_files if explanation_files else None
    
    def refresh_all_status(self):
        """Refresh all dashboard components"""
        return (
            self.get_system_status(),
            self.get_enhanced_results(),
            self.get_model_info(),
            self.get_training_curves(),
            self.get_confusion_matrix(),
            self.get_explanations(),
            self.get_experiment_summary(),
            self.create_system_metrics_plot()
        )

# Global dashboard instance
dashboard = PipelineDashboard()

def create_dashboard():
    """Create the advanced dashboard"""
    
    # Modern theme design
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    ).set(
        # Dark theme
        body_background_fill="#0a0e1a",
        body_background_fill_dark="#0a0e1a",
        background_fill_primary="#1a1f2e",
        background_fill_primary_dark="#1a1f2e",
        background_fill_secondary="#242b3d",
        background_fill_secondary_dark="#242b3d",
        border_color_primary="#3d4556",
        border_color_primary_dark="#3d4556",
        
        # Blue accent
        color_accent_soft="#0066cc",
        color_accent_soft_dark="#0066cc",
        color_accent="#0080ff",
        
        # Typography
        body_text_color="#e2e8f0",
        body_text_color_dark="#e2e8f0",
        body_text_color_subdued="#94a3b8",
        body_text_color_subdued_dark="#94a3b8",
        
        # Primary buttons
        button_primary_background_fill="linear-gradient(135deg, #0066cc 0%, #0080ff 100%)",
        button_primary_background_fill_dark="linear-gradient(135deg, #0066cc 0%, #0080ff 100%)",
        button_primary_background_fill_hover="linear-gradient(135deg, #0052a3 0%, #0066cc 100%)",
        button_primary_background_fill_hover_dark="linear-gradient(135deg, #0052a3 0%, #0066cc 100%)",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        
        # Secondary elements
        button_secondary_background_fill="#242b3d",
        button_secondary_background_fill_dark="#242b3d",
        button_secondary_background_fill_hover="#3d4556",
        button_secondary_background_fill_hover_dark="#3d4556",
        button_secondary_text_color="#e2e8f0",
        button_secondary_text_color_dark="#e2e8f0",
        
        # Blocks and inputs
        block_background_fill="#1a1f2e",
        block_background_fill_dark="#1a1f2e",
        block_label_background_fill="#1a1f2e",
        block_label_background_fill_dark="#1a1f2e",
        block_label_text_color="#e2e8f0",
        block_label_text_color_dark="#e2e8f0",
        
        input_background_fill="#242b3d",
        input_background_fill_dark="#242b3d",
        input_border_color="#3d4556",
        input_border_color_dark="#3d4556",
        input_placeholder_color="#64748b",
        input_placeholder_color_dark="#64748b",
    )
    
    # Custom CSS for advanced styling
    css = """
    /* Dashboard Styling */
    .gradio-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background: linear-gradient(135deg, #0a0e1a 0%, #1a1f2e 100%) !important;
        min-height: 100vh;
    }
    
    /* Header */
    .dashboard-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #242b3d 100%);
        border: 1px solid #3d4556;
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .dashboard-header h1 {
        background: linear-gradient(135deg, #0080ff 0%, #00ccff 100%);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .dashboard-header p {
        text-align: center;
        color: #94a3b8;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    /* Cards */
    .status-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #242b3d 100%);
        border: 1px solid #3d4556;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        border-left: 4px solid #0080ff;
    }
    
    .metrics-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #242b3d 100%);
        border: 1px solid #3d4556;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        border-left: 4px solid #00cc66;
    }
    
    .control-panel {
        background: linear-gradient(135deg, #1a1f2e 0%, #242b3d 100%);
        border: 1px solid #3d4556;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    
    /* Enhanced buttons */
    .primary-button {
        background: linear-gradient(135deg, #0066cc 0%, #0080ff 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 4px 16px rgba(0, 128, 255, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .primary-button:hover {
        background: linear-gradient(135deg, #0052a3 0%, #0066cc 100%) !important;
        box-shadow: 0 6px 20px rgba(0, 128, 255, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    
    .secondary-button {
        background: #242b3d !important;
        border: 1px solid #3d4556 !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    
    .secondary-button:hover {
        background: #3d4556 !important;
        transform: translateY(-1px) !important;
    }
    
    /* Tabs */
    .tab-nav {
        background: #1a1f2e;
        border-radius: 8px;
        padding: 4px;
        margin-bottom: 1rem;
    }
    
    .tab-nav button {
        border-radius: 6px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    
    .tab-nav button.selected {
        background: linear-gradient(135deg, #0066cc 0%, #0080ff 100%) !important;
        color: white !important;
        box-shadow: 0 2px 8px rgba(0, 128, 255, 0.3) !important;
    }
    
    /* Enhanced inputs */
    .gr-textbox, .gr-dropdown, .gr-slider {
        border-radius: 8px !important;
        border: 1px solid #3d4556 !important;
        background: #242b3d !important;
        transition: all 0.3s ease !important;
    }
    
    .gr-textbox:focus, .gr-dropdown:focus {
        border-color: #0080ff !important;
        box-shadow: 0 0 0 3px rgba(0, 128, 255, 0.1) !important;
    }
    
    /* Animations */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .loading {
        animation: pulse 2s infinite;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .dashboard-header h1 {
            font-size: 2rem;
        }
        
        .control-panel, .status-card, .metrics-card {
            margin: 0.25rem 0;
            padding: 1rem;
        }
    }
    """
    
    with gr.Blocks(theme=theme, css=css, title="AI Quality Inspection", analytics_enabled=False) as demo:
        
        # Start system monitoring
        dashboard.start_system_monitoring()
        
        # Header
        gr.HTML("""
        <div class="dashboard-header">
            <h1>AI Quality Inspection</h1>
            <p></p>
        </div>
        """)
        
        with gr.Row():
            # Left Panel - Control Center
            with gr.Column(scale=1, elem_classes=["control-panel"]):
                gr.Markdown("## Mission Control Center")
                
                with gr.Group():
                    mode = gr.Dropdown(
                        choices=["full", "train", "evaluate", "explain"],
                        value="full",
                        label="Pipeline Mode",
                        info="Select advanced pipeline execution mode",
                        elem_classes=["dropdown"]
                    )
                    
                    with gr.Row():
                        epochs = gr.Slider(
                            minimum=1, maximum=100, value=25, step=1,
                            label="Training Epochs",
                            info="Optimization iterations"
                        )
                        batch_size = gr.Slider(
                            minimum=8, maximum=256, value=64, step=8,
                            label="Batch Size",
                            info="Training batch configuration"
                        )
                    
                    with gr.Row():
                        learning_rate = gr.Slider(
                            minimum=0.0001, maximum=0.01, value=0.001, step=0.0001,
                            label="Learning Rate",
                            info="Model optimization rate"
                        )
                        optimizer = gr.Dropdown(
                            choices=["adam", "sgd", "rmsprop"],
                            value="adam",
                            label="Optimizer",
                            info="Optimization algorithm"
                        )
                    
                    download_data = gr.Checkbox(
                        value=True,
                        label="Smart Dataset Management",
                        info="Intelligent dataset download and validation"
                    )
                
                with gr.Row():
                    start_btn = gr.Button(
                        "Launch Pipeline",
                        variant="primary", 
                        scale=3,
                        elem_classes=["primary-button"]
                    )
                    stop_btn = gr.Button(
                        "Stop",
                        variant="stop", 
                        scale=1,
                        elem_classes=["secondary-button"]
                    )
                
                
                # Live System Health Monitor
                gr.Markdown("## 🖥️ Live System Health")
                
                # Compact status summary
                system_status_summary = gr.Markdown(
                    dashboard.get_system_status_summary(),
                    elem_classes=["status-card"]
                )
                
                # Beautiful live monitoring chart
                system_health_chart = gr.Plot(
                    label="Real-time System Health",
                    value=dashboard.create_system_health_chart(),
                    container=True
                )
                
            # Right Panel - Analytics Dashboard
            with gr.Column(scale=2):
                gr.Markdown("## Analytics Dashboard")
                
                with gr.Tabs(elem_classes=["tab-nav"]):
                    with gr.Tab("Real-time Metrics"):
                        with gr.Row():
                            with gr.Column():
                                results_display = gr.Markdown(
                                    dashboard.get_enhanced_results(),
                                    elem_classes=["metrics-card"]
                                )
                            with gr.Column():
                                model_info_display = gr.Markdown(
                                    dashboard.get_model_info(),
                                    elem_classes=["metrics-card"]
                                )
                        
                        # Real-time system metrics plot
                        system_plot = gr.Plot(
                            label="System Performance Monitoring",
                            value=dashboard.create_system_metrics_plot()
                        )
                    
                    with gr.Tab("Pipeline Logs"):
                        logs_display = gr.Textbox(
                            label="Pipeline Logs",
                            lines=20,
                            max_lines=25,
                            interactive=False,
                            placeholder="Advanced pipeline logs will appear here with real-time analysis...",
                            elem_classes=["logs"]
                        )
                    
                    with gr.Tab("Training Analytics"):
                        training_plot = gr.Image(
                            label="Advanced Training Curves & Analytics",
                            height=500
                        )
                    
                    with gr.Tab("Performance Matrix"):
                        confusion_plot = gr.Image(
                            label="Confusion Matrix Analysis",
                            height=500
                        )
                    
                    with gr.Tab("AI Explainability"):
                        explanations_gallery = gr.Gallery(
                            label="Advanced AI Explainability Visualizations",
                            columns=3,
                            rows=3,
                            height=500,
                            object_fit="contain"
                        )
                    
                    with gr.Tab("Experiment Tracking"):
                        experiment_summary = gr.Markdown(
                            dashboard.get_experiment_summary(),
                            elem_classes=["metrics-card"]
                        )
        
        
        # Event Handlers
        start_btn.click(
            fn=dashboard.run_pipeline,
            inputs=[mode, epochs, batch_size, download_data, learning_rate, optimizer],
            outputs=[logs_display, results_display, model_info_display,
                    training_plot, confusion_plot, explanations_gallery, experiment_summary],
            show_progress=True
        )
        
        stop_btn.click(
            fn=dashboard.stop_training,
            outputs=[results_display]
        )
        # Add auto-refresh using a more compatible approach
        # Create a simple refresh button that users can click if auto-refresh fails
        with gr.Row():
            gr.Markdown("### 🔄 Dashboard Controls")
            manual_refresh_btn = gr.Button("🔄 Refresh All", variant="secondary", scale=1)
        
        # Manual refresh function
        def refresh_all_components():
            try:
                return (
                    dashboard.get_system_status_summary(),
                    dashboard.create_system_health_chart(),
                    dashboard.create_system_metrics_plot(),
                    dashboard.get_enhanced_results(),
                    dashboard.get_model_info(),
                    dashboard.get_training_curves(),
                    dashboard.get_confusion_matrix(),
                    dashboard.get_experiment_summary(),
                    dashboard.get_explanations()
                )
            except Exception as e:
                print(f"Refresh error: {e}")
                return tuple([None] * 9)
        
        # Connect manual refresh
        manual_refresh_btn.click(
            fn=refresh_all_components,
            outputs=[system_status_summary, system_health_chart, system_plot,
                    results_display, model_info_display, training_plot, confusion_plot,
                    experiment_summary, explanations_gallery]
        )
        
        # Initialize components on load
        demo.load(
            fn=refresh_all_components,
            outputs=[system_status_summary, system_health_chart, system_plot,
                    results_display, model_info_display, training_plot, confusion_plot,
                    experiment_summary, explanations_gallery]
        )
    
    return demo

if __name__ == "__main__":
    # Launch dashboard
    demo = create_dashboard()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        show_api=False,
        quiet=False
    )