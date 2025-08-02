#!/usr/bin/env python3
"""
Live Dashboard for Explainable AI Quality Inspection Pipeline
Real-time monitoring and interaction with the ML pipeline using Gradio
"""

import gradio as gr
import os
import subprocess
import json
import time
import threading
import queue
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np
import io
import base64

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
        
        self.training_process = None
        self.training_logs = []
        self.is_training = False
        
    def run_pipeline(self, mode, epochs, batch_size, download_data):
        """Run the ML pipeline with specified parameters"""
        try:
            self.is_training = True
            self.training_logs = []
            
            # Build command
            cmd = [
                "python", "main.py",
                "--mode", mode,
                "--epochs", str(epochs),
                "--batch-size", str(batch_size)
            ]
            
            if download_data:
                cmd.append("--download-data")
            
            # Start process
            self.training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Collect output in real-time
            logs = []
            for line in iter(self.training_process.stdout.readline, ''):
                if line:
                    logs.append(line.strip())
                    self.training_logs.append(line.strip())
                    
                    # Yield intermediate updates
                    if len(logs) % 10 == 0:  # Update every 10 lines
                        yield "\\n".join(logs), self.get_training_status(), self.get_current_results()
            
            self.training_process.wait()
            self.is_training = False
            
            # Final update
            yield "\\n".join(logs), "✅ Pipeline completed!", self.get_current_results()
            
        except Exception as e:
            self.is_training = False
            yield f"❌ Error: {str(e)}", "❌ Pipeline failed", ""
    
    def get_training_status(self):
        """Get current training status"""
        if not self.is_training:
            return "🔄 Ready"
        
        # Check for specific patterns in logs
        recent_logs = "\\n".join(self.training_logs[-50:])  # Last 50 lines
        
        if "TRAINING PHASE" in recent_logs:
            return "🚀 Training model..."
        elif "EVALUATION" in recent_logs:
            return "📊 Evaluating model..."
        elif "EXPLAINABILITY" in recent_logs:
            return "🔍 Generating explanations..."
        elif "Downloading" in recent_logs or "download" in recent_logs:
            return "⬇️ Downloading dataset..."
        elif "Found" in recent_logs and "images" in recent_logs:
            return "📁 Loading dataset..."
        else:
            return "⚙️ Processing..."
    
    def get_current_results(self):
        """Get current results and metrics"""
        try:
            # Check for evaluation results
            eval_file = self.logs_dir / "evaluation_results.json"
            if eval_file.exists():
                with open(eval_file, 'r') as f:
                    data = json.load(f)
                    accuracy = data.get('test_accuracy', 0) * 100
                    return f"📈 Test Accuracy: {accuracy:.2f}%"
            
            # Check for training history
            history_file = self.logs_dir / "training_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    if 'val_accuracy' in data and data['val_accuracy']:
                        latest_acc = data['val_accuracy'][-1] * 100
                        return f"📊 Latest Validation Accuracy: {latest_acc:.2f}%"
            
            return "📋 No results yet"
            
        except Exception as e:
            return f"❌ Error reading results: {str(e)}"
    
    def stop_training(self):
        """Stop the current training process"""
        if self.training_process and self.training_process.poll() is None:
            self.training_process.terminate()
            self.is_training = False
            return "🛑 Training stopped", "🔄 Ready"
        return "ℹ️ No active training to stop", "🔄 Ready"
    
    def get_training_curves(self):
        """Get training curves plot"""
        try:
            curves_file = self.logs_dir / "training_curves.png"
            if curves_file.exists():
                return str(curves_file)
            return None
        except:
            return None
    
    def get_confusion_matrix(self):
        """Get confusion matrix plot"""
        try:
            cm_file = self.reports_dir / "confusion_matrix.png"
            if cm_file.exists():
                return str(cm_file)
            return None
        except:
            return None
    
    def get_explanations(self):
        """Get explanation visualizations"""
        try:
            explanations = []
            for i in range(1, 6):  # Check for up to 5 explanation samples
                exp_file = self.reports_dir / f"explanation_sample_{i}.png"
                if exp_file.exists():
                    explanations.append(str(exp_file))
            return explanations if explanations else [None]
        except:
            return [None]
    
    def get_model_info(self):
        """Get model information"""
        try:
            model_file = self.models_dir / "cnn_casting_inspection_model.keras"
            if model_file.exists():
                size_mb = model_file.stat().st_size / (1024 * 1024)
                return f"🤖 Model: CNN (676,945 params)\\n📁 Size: {size_mb:.2f} MB\\n✅ Status: Trained"
            return "🤖 Model: Not trained yet"
        except:
            return "🤖 Model: Status unknown"

def create_dashboard():
    """Create the Gradio dashboard interface"""
    dashboard = PipelineDashboard()
    
    with gr.Blocks(
        title="🔍 Explainable AI Quality Inspection",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .status-box {
            background: linear-gradient(45deg, #1e3c72, #2a5298);
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🔍 Explainable AI Quality Inspection Dashboard
        
        **Real-time monitoring and control of the ML pipeline for industrial defect detection**
        
        🎯 **Features**: CNN Training • Model Evaluation • 4-Method Explainability (LIME, SHAP, Grad-CAM, Integrated Gradients)
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 🚀 Pipeline Control")
                
                with gr.Group():
                    mode = gr.Dropdown(
                        choices=["full", "train", "evaluate", "explain"],
                        value="full",
                        label="🎛️ Pipeline Mode",
                        info="Select pipeline mode to run"
                    )
                    
                    epochs = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=25,
                        step=1,
                        label="🔄 Training Epochs",
                        info="Number of training epochs"
                    )
                    
                    batch_size = gr.Slider(
                        minimum=8,
                        maximum=128,
                        value=64,
                        step=8,
                        label="📦 Batch Size",
                        info="Training batch size"
                    )
                    
                    download_data = gr.Checkbox(
                        value=True,
                        label="⬇️ Download Dataset",
                        info="Download casting dataset if needed"
                    )
                
                with gr.Row():
                    start_btn = gr.Button("🚀 Start Pipeline", variant="primary", scale=2)
                    stop_btn = gr.Button("🛑 Stop", variant="stop", scale=1)
                
                gr.Markdown("## 📊 Status")
                status_display = gr.Textbox(
                    value="🔄 Ready",
                    label="Pipeline Status",
                    interactive=False,
                    elem_classes=["status-box"]
                )
                
                results_display = gr.Textbox(
                    value="📋 No results yet",
                    label="Current Results",
                    interactive=False
                )
                
                model_info = gr.Textbox(
                    value=dashboard.get_model_info(),
                    label="Model Information",
                    interactive=False,
                    every=5
                )
            
            with gr.Column(scale=2):
                gr.Markdown("## 📈 Live Results")
                
                with gr.Tabs():
                    with gr.Tab("📋 Pipeline Logs"):
                        logs_display = gr.Textbox(
                            label="Real-time Logs",
                            lines=20,
                            max_lines=20,
                            autoscroll=True,
                            interactive=False
                        )
                    
                    with gr.Tab("📊 Training Curves"):
                        training_plot = gr.Image(
                            label="Training & Validation Curves",
                            every=10
                        )
                    
                    with gr.Tab("🎯 Confusion Matrix"):
                        confusion_plot = gr.Image(
                            label="Model Performance Matrix",
                            every=10
                        )
                    
                    with gr.Tab("🔍 Explanations"):
                        explanations_gallery = gr.Gallery(
                            label="AI Explainability Visualizations",
                            show_label=True,
                            elem_id="gallery",
                            columns=2,
                            rows=2,
                            every=10
                        )
        
        # Event handlers
        start_output = start_btn.click(
            fn=dashboard.run_pipeline,
            inputs=[mode, epochs, batch_size, download_data],
            outputs=[logs_display, status_display, results_display],
            show_progress=True
        )
        
        stop_btn.click(
            fn=dashboard.stop_training,
            outputs=[status_display, results_display]
        )
        
        # Auto-refresh components
        demo.load(
            fn=dashboard.get_training_curves,
            outputs=training_plot,
            every=10
        )
        
        demo.load(
            fn=dashboard.get_confusion_matrix,
            outputs=confusion_plot,
            every=10
        )
        
        demo.load(
            fn=dashboard.get_explanations,
            outputs=explanations_gallery,
            every=10
        )
        
        gr.Markdown("""
        ---
        
        🔧 **Pipeline Components**: Data Loading • CNN Training • Model Evaluation • Explainability Analysis
        
        📊 **Metrics Tracked**: Accuracy • Precision • Recall • F1-Score • Confusion Matrix • ROC Curves
        
        🎨 **Explainability Methods**: LIME (Local) • SHAP (Global) • Grad-CAM (Attention) • Integrated Gradients (Attribution)
        """)
    
    return demo

if __name__ == "__main__":
    # Create and launch dashboard
    demo = create_dashboard()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )