"""
Visualization utilities for training and analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import cv2

# Set style
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    plt.style.use('seaborn')
sns.set_palette("husl")

def plot_training_history(history, save_path=None, figsize=(15, 5)):
    """
    Plot training history including loss, accuracy, and learning rate.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate plot
    if 'learning_rates' in history:
        axes[2].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
        axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'Learning Rate\nData Not Available', 
                     ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_sample_predictions(model, dataloader, device, num_samples=8, 
                          class_names=None, save_path=None, figsize=(16, 8)):
    """
    Plot sample predictions from the model.
    
    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to run on
        num_samples: Number of samples to show
        class_names: Names of classes
        save_path: Path to save the plot
        figsize: Figure size
    """
    model.eval()
    
    if class_names is None:
        class_names = ['OK', 'Defective']
    
    fig, axes = plt.subplots(2, num_samples//2, figsize=figsize)
    axes = axes.ravel()
    
    samples_shown = 0
    
    with torch.no_grad():
        for batch_idx, (data, target, paths) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            for i in range(data.size(0)):
                if samples_shown >= num_samples:
                    break
                
                # Convert tensor to numpy for visualization
                img = data[i].cpu()
                
                # Denormalize
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img = img * std + mean
                img = torch.clamp(img, 0, 1)
                
                # Convert to HWC format
                img = img.permute(1, 2, 0).numpy()
                
                # Plot
                axes[samples_shown].imshow(img)
                
                # Add prediction information
                true_label = class_names[target[i].item()]
                pred_label = class_names[predictions[i].item()]
                confidence = probabilities[i, predictions[i]].item()
                
                color = 'green' if predictions[i] == target[i] else 'red'
                
                axes[samples_shown].set_title(
                    f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.3f}',
                    color=color, fontweight='bold'
                )
                axes[samples_shown].axis('off')
                
                samples_shown += 1
            
            if samples_shown >= num_samples:
                break
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_feature_maps(model, image, layer_names=None, save_path=None, figsize=(16, 12)):
    """
    Visualize feature maps from different layers.
    
    Args:
        model: Model to extract features from
        image: Input image tensor
        layer_names: Names of layers to visualize
        save_path: Path to save the plot
        figsize: Figure size
    """
    model.eval()
    
    # Get feature maps
    feature_maps = []
    
    def hook_fn(module, input, output):
        feature_maps.append(output.detach())
    
    # Register hooks
    hooks = []
    if hasattr(model, 'backbone') and 'resnet' in str(type(model.backbone)):
        hooks.append(model.backbone.layer1.register_forward_hook(hook_fn))
        hooks.append(model.backbone.layer2.register_forward_hook(hook_fn))
        hooks.append(model.backbone.layer3.register_forward_hook(hook_fn))
        hooks.append(model.backbone.layer4.register_forward_hook(hook_fn))
        
        if layer_names is None:
            layer_names = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4']
    
    # Forward pass
    with torch.no_grad():
        _ = model(image.unsqueeze(0))
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Plot feature maps
    if feature_maps:
        num_layers = len(feature_maps)
        fig, axes = plt.subplots(num_layers, 6, figsize=figsize)
        
        if num_layers == 1:
            axes = axes[np.newaxis, :]
        
        for layer_idx, (fmap, layer_name) in enumerate(zip(feature_maps, layer_names)):
            fmap = fmap.squeeze(0)  # Remove batch dimension
            
            # Select first 6 channels
            num_channels = min(6, fmap.shape[0])
            
            for ch_idx in range(num_channels):
                channel_map = fmap[ch_idx].cpu().numpy()
                
                axes[layer_idx, ch_idx].imshow(channel_map, cmap='viridis')
                axes[layer_idx, ch_idx].set_title(f'{layer_name}\nChannel {ch_idx}')
                axes[layer_idx, ch_idx].axis('off')
            
            # Fill remaining subplots if needed
            for ch_idx in range(num_channels, 6):
                axes[layer_idx, ch_idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    else:
        print("No feature maps captured. Check model architecture.")

def plot_class_distribution(dataset, class_names=None, save_path=None, figsize=(10, 6)):
    """
    Plot class distribution in the dataset.
    
    Args:
        dataset: Dataset object
        class_names: Names of classes
        save_path: Path to save the plot
        figsize: Figure size
    """
    # Count classes
    class_counts = {}
    for _, label, _ in dataset:
        if isinstance(label, torch.Tensor):
            label = label.item()
        class_counts[label] = class_counts.get(label, 0) + 1
    
    if class_names is None:
        class_names = [f'Class {i}' for i in sorted(class_counts.keys())]
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Bar plot
    classes = sorted(class_counts.keys())
    counts = [class_counts[c] for c in classes]
    
    bars = ax1.bar([class_names[c] for c in classes], counts, 
                   color=sns.color_palette("husl", len(classes)))
    ax1.set_title('Class Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Number of Samples')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom')
    
    # Pie chart
    ax2.pie(counts, labels=[class_names[c] for c in classes], autopct='%1.1f%%',
            colors=sns.color_palette("husl", len(classes)))
    ax2.set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_attention_heatmap(attention_weights, image, save_path=None, figsize=(12, 4)):
    """
    Plot attention heatmap overlay on image.
    
    Args:
        attention_weights: Attention weights tensor
        image: Original image
        save_path: Path to save the plot
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Convert tensors to numpy if needed
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image.squeeze(0)
        if image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        image = image.cpu().numpy()
    
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().numpy()
    
    # Ensure image is in [0, 1] range
    if image.max() > 1.0:
        image = image / 255.0
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Attention heatmap
    if len(attention_weights.shape) > 2:
        attention_weights = np.mean(attention_weights, axis=0)
    
    im = axes[1].imshow(attention_weights, cmap='hot', interpolation='bilinear')
    axes[1].set_title('Attention Heatmap')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], shrink=0.8)
    
    # Overlay
    # Resize attention to match image size
    if attention_weights.shape != image.shape[:2]:
        attention_resized = cv2.resize(
            attention_weights, 
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_CUBIC
        )
    else:
        attention_resized = attention_weights
    
    # Normalize attention
    attention_resized = (attention_resized - attention_resized.min()) / \
                      (attention_resized.max() - attention_resized.min())
    
    # Create overlay
    axes[2].imshow(image)
    axes[2].imshow(attention_resized, cmap='jet', alpha=0.5)
    axes[2].set_title('Attention Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_model_summary_plot(model, input_size=(3, 224, 224), save_path=None):
    """
    Create a visual summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
        save_path: Path to save the plot
    """
    try:
        from torchsummary import summary
        
        print("Model Architecture Summary:")
        print("=" * 60)
        summary(model, input_size)
        
    except ImportError:
        print("torchsummary not available. Install with: pip install torchsummary")
        
        # Basic model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")

def plot_gradient_flow(named_parameters, save_path=None, figsize=(12, 6)):
    """
    Plot gradient flow through the network layers.
    
    Args:
        named_parameters: Model named parameters
        save_path: Path to save the plot
        figsize: Figure size
    """
    ave_grads = []
    max_grads = []
    layers = []
    
    for n, p in named_parameters:
        if p.requires_grad and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu().item())
            max_grads.append(p.grad.abs().max().cpu().item())
    
    plt.figure(figsize=figsize)
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.6, label="max-gradient")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.6, label="mean-gradient")
    
    plt.xlabel("Layers")
    plt.ylabel("Gradient")
    plt.title("Gradient Flow")
    plt.legend()
    
    # Rotate layer names for better readability
    plt.xticks(range(0, len(layers), 1), layers[::1], rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()