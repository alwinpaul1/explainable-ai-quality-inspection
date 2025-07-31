"""
Advanced data augmentation strategies specifically designed for casting defect detection
to improve model generalization and reduce overfitting.
"""

import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class CastingDefectAugmentation:
    """
    Advanced augmentation pipeline specifically designed for casting defect detection.
    Focuses on realistic industrial conditions and defect preservation.
    """
    
    def __init__(self, 
                 image_size=(224, 224),
                 augmentation_strength='strong',
                 preserve_defects=True):
        """
        Args:
            image_size: Target image size (height, width)
            augmentation_strength: 'light', 'medium', 'strong', 'progressive'
            preserve_defects: Whether to use defect-preserving augmentations
        """
        self.image_size = image_size
        self.augmentation_strength = augmentation_strength
        self.preserve_defects = preserve_defects
        
    def get_training_transforms(self):
        """Get comprehensive training augmentations for casting defect detection."""
        
        if self.augmentation_strength == 'progressive':
            return self._get_progressive_transforms()
        elif self.augmentation_strength == 'strong':
            return self._get_strong_transforms()
        elif self.augmentation_strength == 'medium':
            return self._get_medium_transforms()
        else:
            return self._get_light_transforms()
    
    def get_validation_transforms(self):
        """Get validation transforms (no augmentation, just normalization)."""
        return A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def get_tta_transforms(self):
        """Get Test Time Augmentation transforms for improved validation accuracy."""
        return [
            # Original
            A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
            # Horizontal flip
            A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
            # Vertical flip
            A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
            # Slight rotation
            A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Rotate(limit=5, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
            # Brightness adjustment
            A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
        ]
    
    def _get_progressive_transforms(self):
        """Progressive augmentation that adapts strength based on epoch."""
        return A.Compose([
            # Preprocessing
            A.Resize(height=int(self.image_size[0] * 1.15), width=int(self.image_size[1] * 1.15)),
            
            # Progressive geometric transformations
            A.OneOf([
                A.RandomCrop(height=self.image_size[0], width=self.image_size[1]),
                A.CenterCrop(height=self.image_size[0], width=self.image_size[1]),
            ], p=1.0),
            
            # Core geometric augmentations (defect-preserving)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.4),
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=25, p=0.6, border_mode=cv2.BORDER_REFLECT),
            
            # Advanced geometric transformations
            A.ShiftScaleRotate(
                shift_limit=0.15, 
                scale_limit=0.25, 
                rotate_limit=20, 
                p=0.6,
                border_mode=cv2.BORDER_REFLECT
            ),
            
            # Industrial lighting conditions simulation
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.4, 
                    contrast_limit=0.4, 
                    p=1.0
                ),
                A.RandomGamma(gamma_limit=(70, 130), p=1.0),
                A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0),
            ], p=0.8),
            
            # Color variations (industrial environment)
            A.HueSaturationValue(
                hue_shift_limit=15, 
                sat_shift_limit=25, 
                val_shift_limit=20, 
                p=0.6
            ),
            
            # Realistic noise simulation
            A.OneOf([
                A.GaussNoise(var_limit=(15.0, 45.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.04), intensity=(0.1, 0.4), p=1.0),
                A.MultiplicativeNoise(multiplier=[0.85, 1.15], p=1.0),
            ], p=0.5),
            
            # Blur variations (camera/motion effects)
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.3),
            
            # Manufacturing environment distortions
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                A.GridDistortion(num_steps=5, distort_limit=0.2, p=1.0),
                A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=1.0),
            ], p=0.4),
            
            # Advanced occlusion simulation
            A.OneOf([
                A.CoarseDropout(
                    max_holes=12, 
                    max_height=24, max_width=24,
                    min_holes=3, 
                    min_height=8, min_width=8,
                    fill_value=0, 
                    p=1.0
                ),
                A.Cutout(
                    num_holes=8, 
                    max_h_size=16, max_w_size=16, 
                    fill_value=0, 
                    p=1.0
                ),
            ], p=0.4),
            
            # Shadow and lighting variations
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=3,
                shadow_dimension=5,
                p=0.3
            ),
            
            # Final normalization
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def _get_strong_transforms(self):
        """Strong augmentation for robust training."""
        return A.Compose([
            # Preprocessing with more variation
            A.Resize(height=int(self.image_size[0] * 1.2), width=int(self.image_size[1] * 1.2)),
            A.RandomCrop(height=self.image_size[0], width=self.image_size[1]),
            
            # Core geometric augmentations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.4),
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=30, p=0.7, border_mode=cv2.BORDER_REFLECT),
            
            # Advanced transformations
            A.ShiftScaleRotate(
                shift_limit=0.2, 
                scale_limit=0.3, 
                rotate_limit=25, 
                p=0.7,
                border_mode=cv2.BORDER_REFLECT
            ),
            
            # Perspective transformation (important for casting inspection)
            A.Perspective(scale=(0.05, 0.15), p=0.4),
            
            # Industrial lighting simulation
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=1.0),
                A.RandomGamma(gamma_limit=(60, 140), p=1.0),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            ], p=0.9),
            
            # Color variations
            A.HueSaturationValue(
                hue_shift_limit=20, 
                sat_shift_limit=30, 
                val_shift_limit=25, 
                p=0.7
            ),
            
            # Channel operations for industrial imagery
            A.OneOf([
                A.ChannelShuffle(p=1.0),
                A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=1.0),
            ], p=0.3),
            
            # Noise simulation
            A.OneOf([
                A.GaussNoise(var_limit=(20.0, 60.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                A.MultiplicativeNoise(multiplier=[0.8, 1.2], p=1.0),
            ], p=0.6),
            
            # Blur and sharpness
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=5, p=1.0),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
            ], p=0.4),
            
            # Distortions
            A.OneOf([
                A.ElasticTransform(alpha=2, sigma=50, alpha_affine=50, p=1.0),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                A.OpticalDistortion(distort_limit=0.15, shift_limit=0.15, p=1.0),
            ], p=0.5),
            
            # Advanced dropout techniques
            A.OneOf([
                A.CoarseDropout(
                    max_holes=15, 
                    max_height=32, max_width=32,
                    min_holes=5, 
                    min_height=8, min_width=8,
                    fill_value=0, 
                    p=1.0
                ),
                A.GridDropout(ratio=0.5, p=1.0),
            ], p=0.5),
            
            # Environmental effects
            A.OneOf([
                A.RandomShadow(p=1.0),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1.0),
            ], p=0.3),
            
            # Normalization
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def _get_medium_transforms(self):
        """Medium strength augmentation."""
        return A.Compose([
            A.Resize(height=int(self.image_size[0] * 1.1), width=int(self.image_size[1] * 1.1)),
            A.RandomCrop(height=self.image_size[0], width=self.image_size[1]),
            
            # Basic geometric
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.2),
            A.Rotate(limit=20, p=0.5),
            
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.2, 
                rotate_limit=15, 
                p=0.5
            ),
            
            # Lighting
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.HueSaturationValue(
                hue_shift_limit=15, 
                sat_shift_limit=20, 
                val_shift_limit=15, 
                p=0.5
            ),
            
            # Noise
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 40.0), p=1.0),
                A.ISONoise(p=1.0),
            ], p=0.4),
            
            # Blur
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.3),
            
            # Dropout
            A.CoarseDropout(
                max_holes=8, 
                max_height=20, max_width=20,
                min_holes=1, 
                min_height=8, min_width=8,
                fill_value=0, 
                p=0.3
            ),
            
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def _get_light_transforms(self):
        """Light augmentation for initial training."""
        return A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            
            # Basic geometric
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=10, p=0.3),
            
            # Basic appearance
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            
            # Light noise
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
            
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


class ProgressiveAugmentationScheduler:
    """
    Scheduler that increases augmentation strength during training to gradually
    make the model more robust.
    """
    
    def __init__(self, total_epochs, start_strength='light', end_strength='strong'):
        self.total_epochs = total_epochs
        self.start_strength = start_strength
        self.end_strength = end_strength
        self.strengths = ['light', 'medium', 'strong']
        
    def get_strength_for_epoch(self, epoch):
        """Get augmentation strength for current epoch."""
        progress = epoch / self.total_epochs
        
        if progress < 0.3:
            return 'light'
        elif progress < 0.7:
            return 'medium'
        else:
            return 'strong'


class MixUpCutMix:
    """
    Advanced mixing strategies for casting defect detection.
    """
    
    def __init__(self, mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        
    def __call__(self, batch):
        data, targets = batch
        
        if np.random.rand() < self.prob:
            if np.random.rand() < 0.5:
                return self.mixup(data, targets)
            else:
                return self.cutmix(data, targets)
        
        return data, targets
    
    def mixup(self, data, targets):
        """Apply MixUp augmentation."""
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = data.size(0)
        index = torch.randperm(batch_size)
        
        mixed_data = lam * data + (1 - lam) * data[index, :]
        targets_a, targets_b = targets, targets[index]
        
        return mixed_data, (targets_a, targets_b, lam)
    
    def cutmix(self, data, targets):
        """Apply CutMix augmentation."""
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        batch_size = data.size(0)
        index = torch.randperm(batch_size)
        
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(data.size(), lam)
        data[:, :, bbx1:bbx2, bby1:bby2] = data[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to exact area ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
        targets_a, targets_b = targets, targets[index]
        
        return data, (targets_a, targets_b, lam)
    
    def rand_bbox(self, size, lam):
        """Generate random bounding box for CutMix."""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2


class DefectPreservingAugmentation:
    """
    Special augmentation techniques that preserve defect characteristics
    while still providing data diversity.
    """
    
    def __init__(self):
        pass
    
    def selective_augmentation(self, image, mask=None):
        """
        Apply augmentation while preserving critical defect areas.
        If mask is provided, it indicates defect regions to preserve.
        """
        # Implementation would depend on having defect masks
        # For now, we use conservative augmentations
        return A.Compose([
            # Conservative geometric transformations
            A.HorizontalFlip(p=0.3),
            A.Rotate(limit=10, p=0.3),
            
            # Careful lighting adjustments
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.5
            ),
            
            # Minimal noise to avoid masking defects
            A.GaussNoise(var_limit=(5.0, 15.0), p=0.2),
        ])


def get_casting_augmentation_pipeline(
    augmentation_type='progressive',
    image_size=(224, 224),
    training_phase='train'
):
    """
    Factory function to get the appropriate augmentation pipeline.
    
    Args:
        augmentation_type: 'light', 'medium', 'strong', 'progressive'
        image_size: Target image size
        training_phase: 'train', 'val', 'test'
    
    Returns:
        Appropriate augmentation pipeline
    """
    
    augmenter = CastingDefectAugmentation(
        image_size=image_size,
        augmentation_strength=augmentation_type,
        preserve_defects=True
    )
    
    if training_phase == 'train':
        return augmenter.get_training_transforms()
    elif training_phase == 'val':
        return augmenter.get_validation_transforms()
    elif training_phase == 'tta':
        return augmenter.get_tta_transforms()
    else:
        return augmenter.get_validation_transforms()