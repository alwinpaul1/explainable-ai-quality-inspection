"""
Dataset utilities for quality inspection following notebook approach
"""

import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

def get_data_generators(data_dir, image_size=(300, 300), batch_size=64, seed=123):
    """
    Create data generators following the notebook approach.
    
    Args:
        data_dir: Root directory containing train and test subdirs
        image_size: Target image size (width, height)
        batch_size: Batch size for training
        seed: Random seed for reproducibility
    
    Returns:
        train_dataset, validation_dataset, test_dataset
    """
    train_path = os.path.join(data_dir, 'train')
    test_path = os.path.join(data_dir, 'test')
    
    # Check if val directory exists, otherwise use test
    val_path = os.path.join(data_dir, 'val')
    if not os.path.exists(val_path):
        val_path = test_path
    
    # Training data generator with augmentation (following notebook)
    train_generator = ImageDataGenerator(
        rotation_range=360,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.75, 1.25],
        rescale=1./255,
        validation_split=0.2
    )
    
    # Test data generator (no augmentation)
    test_generator = ImageDataGenerator(rescale=1./255)
    
    gen_args = dict(
        target_size=image_size,
        color_mode="grayscale",  # Following notebook approach
        batch_size=batch_size,
        class_mode="binary",
        classes={"ok": 0, "defective": 1},
        shuffle=True,
        seed=seed
    )
    
    try:
        train_dataset = train_generator.flow_from_directory(
            directory=train_path,
            subset="training",
            **gen_args
        )
        
        validation_dataset = train_generator.flow_from_directory(
            directory=train_path,
            subset="validation",
            **gen_args
        )
        
        # Use test directory for test dataset
        test_dataset = test_generator.flow_from_directory(
            directory=test_path,
            **gen_args
        )
        
        return train_dataset, validation_dataset, test_dataset
        
    except Exception as e:
        print(f"Error creating data generators: {e}")
        print("Please ensure data structure is:")
        print("  data/")
        print("  ├── train/")
        print("  │   ├── ok/")
        print("  │   └── defective/")
        print("  └── test/")
        print("      ├── ok/")
        print("      └── defective/")
        raise

def analyze_data_distribution(train_dataset, validation_dataset, test_dataset):
    """
    Analyze and display data distribution following notebook approach.
    
    Args:
        train_dataset: Training dataset
        validation_dataset: Validation dataset
        test_dataset: Test dataset
    
    Returns:
        DataFrame with data distribution
    """
    import pandas as pd
    
    # Create data distribution analysis
    image_data = []
    
    for dataset, typ in zip([train_dataset, validation_dataset, test_dataset], 
                           ["train", "validation", "test"]):
        for filename in dataset.filenames:
            class_name = filename.split('/')[0]
            image_data.append({
                "data": typ,
                "class": class_name,
                "filename": filename.split('/')[1]
            })
    
    image_df = pd.DataFrame(image_data)
    data_crosstab = pd.crosstab(
        index=image_df["data"],
        columns=image_df["class"],
        margins=True,
        margins_name="Total"
    )
    
    return data_crosstab

# Legacy function for backward compatibility
def get_data_loaders(data_dir, batch_size=64, num_workers=0, val_split=0.2):
    """
    Get data loaders - now returns TensorFlow datasets.
    
    Args:
        data_dir: Root directory with train/test subdirs
        batch_size: Batch size
        num_workers: Ignored (kept for compatibility)
        val_split: Validation split ratio
    
    Returns:
        train_dataset, validation_dataset (TensorFlow datasets)
    """
    train_dataset, validation_dataset, _ = get_data_generators(
        data_dir=data_dir,
        batch_size=batch_size
    )
    return train_dataset, validation_dataset

def visualize_image_batch(dataset, title, mapping_class={0: "ok", 1: "defect"}):
    """
    Visualize a batch of images following notebook approach.
    
    Args:
        dataset: TensorFlow dataset
        title: Title for the plot
        mapping_class: Class mapping dictionary
    
    Returns:
        Images array
    """
    import matplotlib.pyplot as plt
    
    images, labels = next(iter(dataset))
    batch_size = len(images)
    image_size = images.shape[1:3]
    
    # Determine grid size
    grid_size = int(np.ceil(np.sqrt(batch_size)))
    if batch_size <= 16:
        rows, cols = 4, 4
    else:
        rows = cols = 8
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
    
    for i, (ax, img, label) in enumerate(zip(axes.flat, images, labels)):
        if i >= batch_size:
            ax.axis("off")
            continue
            
        ax.imshow(img.squeeze(), cmap="gray")
        ax.axis("off")
        ax.set_title(mapping_class[int(label)], size=20)
    
    plt.tight_layout()
    fig.suptitle(title, size=30, y=1.05, fontweight="bold")
    plt.show()
    
    return images

class QualityInspectionDataset:
    """
    Quality inspection dataset class for backward compatibility.
    Now wraps TensorFlow dataset functionality.
    """
    
    def __init__(self, data_dir, split='train', transform=None, target_size=(300, 300)):
        """
        Args:
            data_dir: Path to dataset directory
            split: 'train', 'val', or 'test'
            transform: Ignored (kept for compatibility)
            target_size: Target image size
        """
        self.data_dir = data_dir
        self.split = split
        self.target_size = target_size
        
        # Generate TensorFlow datasets
        train_gen, val_gen, test_gen = get_data_generators(
            data_dir, 
            image_size=target_size,
            batch_size=32
        )
        
        if split == 'train':
            self.dataset = train_gen
        elif split == 'val':
            self.dataset = val_gen
        else:
            self.dataset = test_gen
            
        self.classes = ['ok', 'defective']
        self.class_to_idx = {'ok': 0, 'defective': 1}
        
        # Get samples info
        self.samples = self._get_samples_info()
        
    def _get_samples_info(self):
        """Get sample information from TensorFlow dataset."""
        samples = []
        for filename in self.dataset.filenames:
            class_name = filename.split('/')[0]
            full_path = os.path.join(self.data_dir, self.split, filename)
            samples.append((full_path, class_name))
        return samples
        
    def __len__(self):
        return self.dataset.samples
        
    def __getitem__(self, idx):
        """Get item - returns from TensorFlow dataset."""
        # This is a simplified version for compatibility
        # In practice, you would iterate through the TensorFlow dataset
        images, labels = next(iter(self.dataset))
        if idx < len(images):
            return images[idx], labels[idx], f"sample_{idx}"
        else:
            raise IndexError("Index out of range")