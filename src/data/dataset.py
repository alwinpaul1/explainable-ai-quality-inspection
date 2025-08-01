"""
Dataset utilities for quality inspection
"""

import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

def get_data_generators(data_dir, image_size=(300, 300), batch_size=64, seed=123):
    """
    Create data generators for casting product quality inspection.
    
    Args:
        data_dir: Root directory containing the casting dataset
        image_size: Target image size (width, height)
        batch_size: Batch size for training
        seed: Random seed for reproducibility
    
    Returns:
        train_dataset, validation_dataset, test_dataset
    """
    # Use casting_data/casting_data/ as main directory structure
    my_data_dir = os.path.join(data_dir, 'casting_data', 'casting_data')
    train_path = os.path.join(my_data_dir, 'train')
    test_path = os.path.join(my_data_dir, 'test')
    
    print(f"📁 Using data directory: {my_data_dir}")
    print(f"📁 Train path: {train_path}")
    print(f"📁 Test path: {test_path}")
    
    # Production constants
    IMAGE_SIZE = image_size
    BATCH_SIZE = batch_size
    SEED_NUMBER = seed
    
    # Training data generator with augmentation
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
    
    # Generator arguments
    gen_args = dict(
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="binary",
        classes={"ok_front": 0, "def_front": 1},
        shuffle=True,
        seed=SEED_NUMBER
    )
    
    try:
        # Create datasets
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

def analyze_data_distribution(train_dataset, validation_dataset, test_dataset, save_plots=True, save_dir='results/reports'):
    """
    Analyze and display data distribution following notebook approach.
    
    Args:
        train_dataset: Training dataset
        validation_dataset: Validation dataset
        test_dataset: Test dataset
        save_plots: Whether to save visualization plots
        save_dir: Directory to save plots
    
    Returns:
        DataFrame with data distribution
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
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
    
    print("📊 IMAGE DATA PROPORTION:")
    print(data_crosstab)
    
    # Create visualization following notebook style
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)
        
        total_image = data_crosstab.iloc[-1, -1]
        ax = data_crosstab.iloc[:-1, :-1].T.plot(kind="bar", stacked=True, rot=0, figsize=(10, 6))
        
        percent_val = []
        
        for rect in ax.patches:
            height = rect.get_height()
            width = rect.get_width()
            percent = 100 * height / total_image
            
            ax.text(rect.get_x() + width - 0.25, 
                    rect.get_y() + height/2, 
                    int(height), 
                    ha='center',
                    va='center',
                    color="white",
                    fontsize=10)
            
            ax.text(rect.get_x() + width + 0.01, 
                    rect.get_y() + height/2, 
                    "{:.2f}%".format(percent), 
                    ha='left',
                    va='center',
                    color="black",
                    fontsize=10)
            
            percent_val.append(percent)
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels)
        
        percent_def = sum(percent_val[::2])
        ax.set_xticklabels([f"def_front\n({percent_def:.2f} %)", f"ok_front\n({100-percent_def:.2f} %)"])
        plt.title("IMAGE DATA PROPORTION", fontsize=15, fontweight="bold")
        
        # Save plot
        prop_path = os.path.join(save_dir, 'data_proportion.png')
        plt.savefig(prop_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Data proportion visualization saved: {prop_path}")
    
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
    Visualize a batch of images.
    
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