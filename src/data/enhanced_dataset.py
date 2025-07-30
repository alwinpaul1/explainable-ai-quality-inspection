"""
Enhanced dataset with advanced augmentation techniques
"""

import os
import torch
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2

class EnhancedQualityInspectionDataset(Dataset):
    """Enhanced dataset with advanced augmentation techniques."""
    
    def __init__(self, data_dir, split='train', transform=None, target_size=(224, 224), 
                 use_advanced_aug=True):
        """
        Args:
            data_dir: Path to dataset directory
            split: 'train', 'val', or 'test'
            transform: Optional transforms to apply
            target_size: Target image size (height, width)
            use_advanced_aug: Use advanced augmentation techniques
        """
        self.data_dir = data_dir
        self.split = split
        self.target_size = target_size
        self.use_advanced_aug = use_advanced_aug
        
        # Load dataset
        self.samples = self._load_samples()
        self.classes = self._get_classes()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Setup transforms
        if transform is None:
            self.transform = self._get_default_transforms()
        else:
            self.transform = transform
            
        # Setup advanced augmentation
        if use_advanced_aug and split == 'train':
            self.advanced_transform = self._get_advanced_transforms()
        else:
            self.advanced_transform = None
        
    def _load_samples(self):
        """Load image paths and labels with better error handling."""
        samples = []
        
        # For organized dataset structure
        if self.split in ['train', 'val', 'test']:
            base_dir = os.path.join(self.data_dir, self.split)
        else:
            base_dir = self.data_dir
            
        if not os.path.exists(base_dir):
            print(f"Warning: Directory {base_dir} not found, trying alternative structure...")
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
            # Standard organized structure
            for class_dir in os.listdir(base_dir):
                class_path = os.path.join(base_dir, class_dir)
                if os.path.isdir(class_path):
                    for img_file in os.listdir(class_path):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(class_path, img_file)
                            samples.append((img_path, class_dir))
        
        print(f"Loaded {len(samples)} samples for {self.split} split")
        return samples
    
    def _get_classes(self):
        """Get unique class names."""
        classes = list(set([sample[1] for sample in self.samples]))
        return sorted(classes)
    
    def _get_default_transforms(self):
        """Get default transforms based on split."""
        if self.split == 'train':
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(self.target_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def _get_advanced_transforms(self):
        """Get advanced albumentations transforms for training."""
        return A.Compose([
            A.Resize(256, 256),
            A.RandomCrop(height=self.target_size[0], width=self.target_size[1]),
            
            # Geometric augmentations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=20, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
            
            # Appearance augmentations
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
                A.MultiplicativeNoise(multiplier=[0.9, 1.1]),
            ], p=0.4),
            
            A.OneOf([
                A.MotionBlur(blur_limit=3),
                A.MedianBlur(blur_limit=3),
                A.GaussianBlur(blur_limit=3),
            ], p=0.3),
            
            # Distortions specific to manufacturing defects
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50),
                A.GridDistortion(num_steps=5, distort_limit=0.3),
                A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1),
            ], p=0.3),
            
            # Cutout/Coarse dropout for robustness
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, 
                           min_holes=1, min_height=8, min_width=8, 
                           fill_value=0, p=0.3),
            
            # Normalization
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image with better error handling
        try:
            image = Image.open(img_path).convert('RGB')
            # Ensure minimum size
            if min(image.size) < 64:
                image = image.resize((224, 224), Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image if loading fails
            image = Image.new('RGB', self.target_size, color=(0, 0, 0))
        
        # Apply advanced augmentation if available
        if self.advanced_transform is not None:
            # Convert PIL to numpy for albumentations
            image_np = np.array(image)
            augmented = self.advanced_transform(image=image_np)
            image_tensor = augmented['image']
        else:
            # Apply standard transforms
            image_tensor = self.transform(image)
        
        # Convert label to index
        label_idx = self.class_to_idx[label]
        
        return image_tensor, label_idx, img_path

class BalancedDataLoader:
    """Create balanced data loaders with proper sampling."""
    
    def __init__(self, data_dir, batch_size=32, num_workers=4, val_split=0.2, 
                 use_advanced_aug=True, use_stratified_split=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.use_advanced_aug = use_advanced_aug
        self.use_stratified_split = use_stratified_split
        
    def get_balanced_loaders(self):
        """Create balanced train and validation data loaders."""
        
        # Create datasets
        try:
            train_dataset = EnhancedQualityInspectionDataset(
                self.data_dir, split='train', use_advanced_aug=self.use_advanced_aug
            )
            
            # Try to load separate test set
            try:
                val_dataset = EnhancedQualityInspectionDataset(
                    self.data_dir, split='test', use_advanced_aug=False
                )
                print("Using separate test set for validation")
            except:
                # Create validation split from training data
                if self.use_stratified_split:
                    train_dataset, val_dataset = self._create_stratified_split(train_dataset)
                else:
                    val_dataset = self._create_random_split(train_dataset)
                print("Created validation split from training data")
        
        except Exception as e:
            print(f"Error creating datasets: {e}")
            print("Creating dummy datasets for testing...")
            train_dataset = DummyEnhancedDataset(size=1000, use_advanced_aug=self.use_advanced_aug)
            val_dataset = DummyEnhancedDataset(size=200, use_advanced_aug=False)
        
        # Calculate class weights for balanced sampling
        train_class_weights = self._calculate_class_weights(train_dataset)
        
        # Create weighted sampler for balanced training
        sample_weights = [train_class_weights[sample[1]] for sample in train_dataset.samples]
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            sampler=sampler,
            num_workers=self.num_workers, 
            pin_memory=True,
            drop_last=True  # For consistent batch sizes
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers, 
            pin_memory=True
        )
        
        # Print dataset statistics
        self._print_dataset_stats(train_dataset, val_dataset)
        
        return train_loader, val_loader
    
    def _create_stratified_split(self, dataset):
        """Create stratified train/validation split."""
        # Get labels for stratification
        labels = [sample[1] for sample in dataset.samples]
        label_indices = [dataset.class_to_idx[label] for label in labels]
        
        # Create stratified split
        from sklearn.model_selection import train_test_split
        
        train_indices, val_indices = train_test_split(
            range(len(dataset.samples)),
            test_size=self.val_split,
            stratify=label_indices,
            random_state=42
        )
        
        # Create subset datasets
        train_samples = [dataset.samples[i] for i in train_indices]
        val_samples = [dataset.samples[i] for i in val_indices]
        
        # Create new dataset instances
        train_dataset = EnhancedQualityInspectionDataset(
            self.data_dir, split='train', use_advanced_aug=self.use_advanced_aug
        )
        train_dataset.samples = train_samples
        
        val_dataset = EnhancedQualityInspectionDataset(
            self.data_dir, split='val', use_advanced_aug=False
        )
        val_dataset.samples = val_samples
        val_dataset.classes = train_dataset.classes
        val_dataset.class_to_idx = train_dataset.class_to_idx
        
        return train_dataset, val_dataset
    
    def _calculate_class_weights(self, dataset):
        """Calculate class weights for balanced sampling."""
        class_counts = {}
        for sample in dataset.samples:
            label = sample[1]
            class_counts[label] = class_counts.get(label, 0) + 1
        
        total_samples = sum(class_counts.values())
        class_weights = {}
        
        for class_name, count in class_counts.items():
            weight = total_samples / (len(class_counts) * count)
            class_weights[class_name] = weight
        
        print(f"Class weights: {class_weights}")
        return class_weights
    
    def _print_dataset_stats(self, train_dataset, val_dataset):
        """Print dataset statistics."""
        print("\n📊 DATASET STATISTICS")
        print("=" * 40)
        
        # Training set stats
        train_class_counts = {}
        for sample in train_dataset.samples:
            label = sample[1]
            train_class_counts[label] = train_class_counts.get(label, 0) + 1
        
        print(f"Training set: {len(train_dataset)} samples")
        for class_name, count in train_class_counts.items():
            percentage = count / len(train_dataset) * 100
            print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
        
        # Validation set stats
        val_class_counts = {}
        for sample in val_dataset.samples:
            label = sample[1]
            val_class_counts[label] = val_class_counts.get(label, 0) + 1
        
        print(f"Validation set: {len(val_dataset)} samples")
        for class_name, count in val_class_counts.items():
            percentage = count / len(val_dataset) * 100
            print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
        
        print("=" * 40)

class DummyEnhancedDataset(Dataset):
    """Enhanced dummy dataset for testing."""
    
    def __init__(self, size=100, use_advanced_aug=False, num_classes=2):
        self.size = size
        self.num_classes = num_classes
        self.classes = ['ok', 'defective']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = [(f"dummy_{i}.jpg", self.classes[i % num_classes]) for i in range(size)]
        
        if use_advanced_aug:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate random image
        image_np = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        if hasattr(self.transform, 'transforms'):  # Albumentations
            augmented = self.transform(image=image_np)
            image = augmented['image']
        else:  # torchvision
            image = Image.fromarray(image_np)
            image = self.transform(image)
        
        # Get label
        label = idx % self.num_classes
        path = f"dummy_{idx}.jpg"
        
        return image, label, path

# Convenience function for external use
def get_enhanced_data_loaders(data_dir, batch_size=32, num_workers=4, val_split=0.2, 
                             use_advanced_aug=True, use_stratified_split=True):
    """Get enhanced data loaders with all improvements."""
    loader_creator = BalancedDataLoader(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=val_split,
        use_advanced_aug=use_advanced_aug,
        use_stratified_split=use_stratified_split
    )
    
    return loader_creator.get_balanced_loaders()