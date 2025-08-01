"""
Training script for quality inspection model
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.dataset import get_data_generators, analyze_data_distribution
from src.models.cnn_model import create_simple_cnn

class QualityInspectionTrainer:
    """Trainer class for quality inspection models."""
    
    def __init__(self, config):
        self.config = config
        
        # Create directories
        os.makedirs(config['save_dir'], exist_ok=True)
        os.makedirs(config['log_dir'], exist_ok=True)
        
        # Initialize model architecture
        self.model = create_simple_cnn(
            input_shape=(300, 300, 1),  # Grayscale images
            num_classes=1  # Binary classification with sigmoid
        )
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model Architecture:")
        self.model.summary()
        
        # Training history
        self.history = None
        self.best_val_acc = 0.0
        
    def train(self, train_dataset, validation_dataset):
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset generator
            validation_dataset: Validation dataset generator
            
        Returns:
            Path to saved best model
        """
        print("\n" + "="*60)
        print("🚀 TRAINING PHASE")
        print("="*60)
        
        # Training parameters
        epochs = self.config.get('epochs', 25)
        steps_per_epoch = self.config.get('steps_per_epoch', 150)
        validation_steps = self.config.get('validation_steps', 150)
        
        print(f"Training Configuration:")
        print(f"- Epochs: {epochs}")
        print(f"- Steps per epoch: {steps_per_epoch}")
        print(f"- Validation steps: {validation_steps}")
        print(f"- Images per epoch: {steps_per_epoch * train_dataset.batch_size}")
        
        # Setup model checkpoint
        model_path = os.path.join(self.config['save_dir'], 'cnn_casting_inspection_model.hdf5')
        checkpoint = ModelCheckpoint(
            model_path,
            verbose=1,
            save_best_only=True,
            monitor='val_loss'
        )
        
        # Train model
        print(f"\n🔥 Starting training...")
        self.history = self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_steps=validation_steps,
            callbacks=[checkpoint],
            verbose=1
        )
        
        # Save training history
        history_path = os.path.join(self.config['log_dir'], 'training_history.json')
        with open(history_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            history_dict = {}
            for key, values in self.history.history.items():
                history_dict[key] = [float(v) for v in values]
            json.dump(history_dict, f, indent=2)
        
        # Plot training curves
        self._plot_training_curves()
        
        # Get best validation accuracy
        self.best_val_acc = max(self.history.history['val_accuracy']) * 100
        
        print(f"\n🎉 Training completed!")
        print(f"📁 Best model saved: {model_path}")
        print(f"📈 Best validation accuracy: {self.best_val_acc:.2f}%")
        
        return model_path
    
    def _plot_training_curves(self):
        """Plot training curves."""
        if self.history is None:
            return
            
        # Create training evaluation plot
        plt.figure(figsize=(8, 6))
        
        # Convert history to DataFrame for seaborn
        history_df = pd.DataFrame(
            self.history.history,
            index=range(1, 1 + len(self.history.epoch))
        )
        
        # Plot with seaborn styling
        sns.lineplot(data=history_df)
        plt.title("TRAINING EVALUATION", fontweight="bold", fontsize=20)
        plt.xlabel("Epochs")
        plt.ylabel("Metrics")
        
        # Update legend labels
        plt.legend(labels=['val loss', 'val accuracy', 'train loss', 'train accuracy'])
        
        # Save plot
        plot_path = os.path.join(self.config['log_dir'], 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Training curves saved: {plot_path}")
    
    def evaluate_on_test(self, test_dataset, threshold=0.5):
        """
        Evaluate model on test data.
        
        Args:
            test_dataset: Test dataset generator
            threshold: Classification threshold
            
        Returns:
            Evaluation results dictionary
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        print("\n" + "="*60)
        print("📊 TESTING ON UNSEEN IMAGES")
        print("="*60)
        
        # Make predictions
        print("Making predictions on test dataset...")
        y_pred_prob = self.model.predict(test_dataset, verbose=1)
        
        # Convert probabilities to classes using threshold
        y_pred_class = (y_pred_prob >= threshold).reshape(-1,)
        y_true_class = test_dataset.classes[test_dataset.index_array]
        
        # Create confusion matrix
        cm = confusion_matrix(y_true_class, y_pred_class)
        cm_df = pd.DataFrame(
            cm,
            index=[["Actual", "Actual"], ["ok", "defect"]],
            columns=[["Predicted", "Predicted"], ["ok", "defect"]]
        )
        
        print("\nConfusion Matrix:")
        print(cm_df)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true_class, y_pred_class, digits=4))
        
        # Visualize predictions
        self._visualize_predictions(test_dataset, threshold)
        
        return {
            'confusion_matrix': cm,
            'y_true': y_true_class,
            'y_pred': y_pred_class,
            'y_pred_prob': y_pred_prob
        }
    
    def _visualize_predictions(self, test_dataset, threshold=0.5):
        """Visualize predictions."""
        mapping_class = {0: "ok", 1: "defect"}
        
        # Get a batch for visualization
        images, labels = next(iter(test_dataset))
        batch_size = min(16, len(images))  # Show up to 16 images
        
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        
        for ax, img, label in zip(axes.flat, images[:batch_size], labels[:batch_size]):
            ax.imshow(img.squeeze(), cmap="gray")
            true_label = mapping_class[int(label)]
            
            # Make prediction for this image
            pred_prob = self.model.predict(img.reshape(1, 300, 300, 1), verbose=0)[0][0]
            pred_label = mapping_class[int(pred_prob >= threshold)]
            
            prob_class = 100 * pred_prob if pred_label == "defect" else 100 * (1 - pred_prob)
            
            ax.set_title(f"TRUE LABEL: {true_label}", fontweight="bold", fontsize=18)
            ax.set_xlabel(
                f"PREDICTED LABEL: {pred_label}\nProb({pred_label}) = {prob_class:.2f}%",
                fontweight="bold", 
                fontsize=15,
                color="blue" if true_label == pred_label else "red"
            )
            
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        fig.suptitle(
            "TRUE VS PREDICTED LABEL FOR 16 RANDOM TEST IMAGES", 
            size=30, y=1.03, fontweight="bold"
        )
        
        # Save visualization
        viz_path = os.path.join(self.config['log_dir'], 'test_predictions.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Prediction visualization saved: {viz_path}")

def train_model(config):
    """
    Train the quality inspection model.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        Trained model path
    """
    print("\n" + "="*70)
    print("🚀 QUALITY INSPECTION TRAINING")
    print("="*70)
    
    # Create data generators
    try:
        train_dataset, validation_dataset, test_dataset = get_data_generators(
            data_dir=config['data_dir'],
            image_size=(300, 300),
            batch_size=config.get('batch_size', 64),
            seed=123
        )
        
        print("✅ Data generators created successfully!")
        
        # Analyze data distribution
        data_crosstab = analyze_data_distribution(train_dataset, validation_dataset, test_dataset)
        print("\nData Distribution:")
        print(data_crosstab)
        
        # Visualize sample images
        from src.data.dataset import visualizeImageBatch, visualize_detailed_image
        train_images = visualizeImageBatch(
            train_dataset, 
            "FIRST BATCH OF THE TRAINING IMAGES\n(WITH DATA AUGMENTATION)"
        )
        
        # Visualize detailed image pixels
        print("Visualizing detailed image pixels (25x25 window):")
        visualize_detailed_image(train_images, image_index=4, start_pixel=(75, 75), size=25)
        
    except Exception as e:
        print(f"❌ Error creating data generators: {e}")
        return None
    
    # Initialize trainer
    trainer = QualityInspectionTrainer(config)
    
    # Train model
    best_model_path = trainer.train(train_dataset, validation_dataset)
    
    # Evaluate on test data
    if test_dataset is not None:
        results = trainer.evaluate_on_test(test_dataset)
        
        # Save evaluation results
        eval_path = os.path.join(config['log_dir'], 'evaluation_results.json')
        eval_data = {
            'confusion_matrix': results['confusion_matrix'].tolist(),
            'test_accuracy': float(np.mean(results['y_true'] == results['y_pred'])),
            'threshold': 0.5
        }
        
        with open(eval_path, 'w') as f:
            json.dump(eval_data, f, indent=2)
        
        print(f"📊 Evaluation results saved: {eval_path}")
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETED!")
    print("="*70)
    
    return best_model_path