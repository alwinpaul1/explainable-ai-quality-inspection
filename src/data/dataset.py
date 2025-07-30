"""
Dataset utilities for quality inspection
"""

import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

class QualityInspectionDataset(Dataset):
    """Dataset class for quality inspection images."""
    
    def __init__(self, data_dir, split='train', transform=None, target_size=(224, 224)):
        """
        Args:
            data_dir: Path to dataset directory
            split: 'train', 'val', or 'test'
            transform: Optional transforms to apply
            target_size: Target image size (height, width)
        """
        self.data_dir = data_dir
        self.split = split
        self.target_size = target_size
        
        # Default transforms if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        # Load dataset
        self.samples = self._load_samples()
        self.classes = self._get_classes()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def _load_samples(self):
        """Load image paths and labels."""
        samples = []
        
        # For casting dataset structure
        train_dir = os.path.join(self.data_dir, 'train')
        test_dir = os.path.join(self.data_dir, 'test')
        
        if self.split == 'train':
            base_dir = train_dir
        else:
            base_dir = test_dir
            
        if not os.path.exists(base_dir):
            # Try alternative structure
            for class_dir in ['def_front', 'ok_front']:
                class_path = os.path.join(self.data_dir, class_dir)
                if os.path.exists(class_path):
                    for img_file in os.listdir(class_path):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(class_path, img_file)
                            label = 'defective' if 'def' in class_dir else 'ok'
                            samples.append((img_path, label))
        else:
            # Standard train/test structure
            for class_dir in os.listdir(base_dir):
                class_path = os.path.join(base_dir, class_dir)
                if os.path.isdir(class_path):
                    for img_file in os.listdir(class_path):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(class_path, img_file)
                            samples.append((img_path, class_dir))
        
        return samples
    
    def _get_classes(self):
        """Get unique class names."""
        classes = list(set([sample[1] for sample in self.samples]))
        return sorted(classes)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image if loading fails
            image = Image.new('RGB', self.target_size, color=(0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Convert label to index
        label_idx = self.class_to_idx[label]
        
        return image, label_idx, img_path

def get_data_loaders(data_dir, batch_size=32, num_workers=4, val_split=0.2):
    """Create train and validation data loaders."""
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Standard transform for validation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    try:
        train_dataset = QualityInspectionDataset(
            data_dir, split='train', transform=train_transform
        )
        
        # Try to load test set, if not available, split train set
        try:
            val_dataset = QualityInspectionDataset(
                data_dir, split='test', transform=val_transform
            )
        except:
            # Split train dataset for validation
            total_size = len(train_dataset)
            val_size = int(total_size * val_split)
            train_size = total_size - val_size
            
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
            
            # Apply validation transform to val_dataset
            val_dataset.dataset.transform = val_transform
    
    except Exception as e:
        print(f"Error creating datasets: {e}")
        print("Creating dummy datasets for testing...")
        
        # Create dummy datasets for testing
        train_dataset = DummyDataset(size=100, transform=train_transform)
        val_dataset = DummyDataset(size=20, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader

class DummyDataset(Dataset):
    """Dummy dataset for testing when real data is not available."""
    
    def __init__(self, size=100, transform=None, num_classes=2):
        self.size = size
        self.transform = transform
        self.num_classes = num_classes
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate random image
        image = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        
        if self.transform:
            image = self.transform(image)
        
        # Random label
        label = np.random.randint(0, self.num_classes)
        
        return image, label, f"dummy_{idx}.jpg"