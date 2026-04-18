"""Data processing and dataset classes for face authentication."""

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class FaceDataset(Dataset):
    """Dataset class for face authentication data."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        image_size: int = 224,
        augmentation: bool = True,
        normalize: bool = True
    ):
        """Initialize face dataset.
        
        Args:
            data_dir: Directory containing face data
            split: Dataset split (train, val, test)
            image_size: Target image size
            augmentation: Whether to apply data augmentation
            normalize: Whether to normalize images
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Setup transforms
        self.transform = self._get_transforms(augmentation, normalize)
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load dataset metadata.
        
        Returns:
            DataFrame containing image paths and labels
        """
        metadata_file = self.data_dir / f"{self.split}_metadata.csv"
        
        if metadata_file.exists():
            return pd.read_csv(metadata_file)
        else:
            # Generate synthetic metadata if file doesn't exist
            return self._generate_synthetic_metadata()
    
    def _generate_synthetic_metadata(self) -> pd.DataFrame:
        """Generate synthetic metadata for demonstration.
        
        Returns:
            DataFrame with synthetic metadata
        """
        # Create synthetic data structure
        users = [f"user_{i:03d}" for i in range(50)]
        images_per_user = 5
        
        data = []
        for user_id, user in enumerate(users):
            for img_idx in range(images_per_user):
                data.append({
                    'image_path': f"synthetic/{user}/image_{img_idx:02d}.jpg",
                    'user_id': user_id,
                    'user_name': user,
                    'is_enrollment': img_idx == 0,  # First image is enrollment
                    'liveness_label': 1,  # All synthetic images are "real"
                    'quality_score': random.uniform(0.7, 1.0)
                })
        
        return pd.DataFrame(data)
    
    def _get_transforms(
        self, 
        augmentation: bool, 
        normalize: bool
    ) -> transforms.Compose:
        """Get image transforms.
        
        Args:
            augmentation: Whether to apply augmentation
            normalize: Whether to normalize images
            
        Returns:
            Composed transforms
        """
        transform_list = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()
        ]
        
        if augmentation and self.split == "train":
            transform_list.insert(-1, transforms.RandomHorizontalFlip(p=0.5))
            transform_list.insert(-1, transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ))
            transform_list.insert(-1, transforms.RandomRotation(15))
        
        if normalize:
            transform_list.append(transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ))
        
        return transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item.
        
        Args:
            idx: Item index
            
        Returns:
            Dictionary containing image and labels
        """
        row = self.metadata.iloc[idx]
        
        # Load image
        image_path = self.data_dir / row['image_path']
        
        if image_path.exists():
            image = Image.open(image_path).convert('RGB')
        else:
            # Generate synthetic image if file doesn't exist
            image = self._generate_synthetic_image()
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        return {
            'image': image_tensor,
            'user_id': torch.tensor(row['user_id'], dtype=torch.long),
            'user_name': row['user_name'],
            'is_enrollment': torch.tensor(row['is_enrollment'], dtype=torch.bool),
            'liveness_label': torch.tensor(row['liveness_label'], dtype=torch.long),
            'quality_score': torch.tensor(row['quality_score'], dtype=torch.float),
            'image_path': str(image_path)
        }
    
    def _generate_synthetic_image(self) -> Image.Image:
        """Generate synthetic face image for demonstration.
        
        Returns:
            PIL Image with synthetic face
        """
        # Create a simple synthetic face-like image
        img = np.random.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # Add some face-like features
        center_x, center_y = self.image_size // 2, self.image_size // 2
        
        # Face outline (oval)
        cv2.ellipse(img, (center_x, center_y), (80, 100), 0, 0, 360, (200, 180, 160), -1)
        
        # Eyes
        cv2.circle(img, (center_x - 30, center_y - 20), 8, (0, 0, 0), -1)
        cv2.circle(img, (center_x + 30, center_y - 20), 8, (0, 0, 0), -1)
        
        # Nose
        cv2.ellipse(img, (center_x, center_y), (5, 15), 0, 0, 360, (150, 130, 110), -1)
        
        # Mouth
        cv2.ellipse(img, (center_x, center_y + 30), (20, 8), 0, 0, 180, (100, 50, 50), -1)
        
        return Image.fromarray(img)


class FaceDataProcessor:
    """Data processor for face authentication system."""
    
    def __init__(
        self,
        image_size: int = 224,
        normalize: bool = True
    ):
        """Initialize data processor.
        
        Args:
            image_size: Target image size
            normalize: Whether to normalize images
        """
        self.image_size = image_size
        self.normalize = normalize
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ) if normalize else transforms.Lambda(lambda x: x)
        ])
    
    def preprocess_image(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        """Preprocess a single image.
        
        Args:
            image: Image path or PIL Image
            
        Returns:
            Preprocessed image tensor
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        return self.transform(image)
    
    def detect_faces(self, image: Union[str, Path, Image.Image]) -> List[Dict[str, Any]]:
        """Detect faces in image.
        
        Args:
            image: Input image
            
        Returns:
            List of detected face regions
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        # Convert to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Load face cascade
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Detect faces
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Convert to list of dictionaries
        face_regions = []
        for (x, y, w, h) in faces:
            face_regions.append({
                'bbox': (x, y, w, h),
                'confidence': 1.0,  # Cascade classifier doesn't provide confidence
                'landmarks': None  # Could be extended with facial landmarks
            })
        
        return face_regions
    
    def extract_face_region(
        self, 
        image: Union[str, Path, Image.Image], 
        face_region: Dict[str, Any]
    ) -> Image.Image:
        """Extract face region from image.
        
        Args:
            image: Input image
            face_region: Face region information
            
        Returns:
            Extracted face image
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        x, y, w, h = face_region['bbox']
        
        # Add some padding
        padding = 0.2
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        x = max(0, x - pad_w)
        y = max(0, y - pad_h)
        w = min(image.width - x, w + 2 * pad_w)
        h = min(image.height - y, h + 2 * pad_h)
        
        # Crop face region
        face_image = image.crop((x, y, x + w, y + h))
        
        return face_image
    
    def create_data_splits(
        self,
        data_dir: Union[str, Path],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ) -> Dict[str, pd.DataFrame]:
        """Create train/validation/test splits.
        
        Args:
            data_dir: Directory containing data
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing split metadata
        """
        random.seed(random_seed)
        
        # Load all metadata
        all_data = []
        data_dir = Path(data_dir)
        
        # Scan for image files
        for image_path in data_dir.rglob("*.jpg"):
            if image_path.is_file():
                # Extract user information from path
                user_name = image_path.parent.name
                user_id = hash(user_name) % 1000  # Simple hash-based user ID
                
                all_data.append({
                    'image_path': str(image_path.relative_to(data_dir)),
                    'user_id': user_id,
                    'user_name': user_name,
                    'is_enrollment': 'enrollment' in str(image_path),
                    'liveness_label': 1,  # Assume all are real for now
                    'quality_score': random.uniform(0.7, 1.0)
                })
        
        df = pd.DataFrame(all_data)
        
        # Create splits by user (to avoid data leakage)
        unique_users = df['user_id'].unique()
        random.shuffle(unique_users)
        
        n_users = len(unique_users)
        n_train = int(n_users * train_ratio)
        n_val = int(n_users * val_ratio)
        
        train_users = unique_users[:n_train]
        val_users = unique_users[n_train:n_train + n_val]
        test_users = unique_users[n_train + n_val:]
        
        # Create splits
        splits = {
            'train': df[df['user_id'].isin(train_users)],
            'val': df[df['user_id'].isin(val_users)],
            'test': df[df['user_id'].isin(test_users)]
        }
        
        # Save splits
        for split_name, split_df in splits.items():
            output_path = data_dir / f"{split_name}_metadata.csv"
            split_df.to_csv(output_path, index=False)
        
        return splits


def create_data_loaders(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    augmentation: bool = True
) -> Dict[str, DataLoader]:
    """Create data loaders for all splits.
    
    Args:
        data_dir: Directory containing data
        batch_size: Batch size
        num_workers: Number of worker processes
        image_size: Target image size
        augmentation: Whether to apply augmentation
        
    Returns:
        Dictionary containing data loaders
    """
    data_loaders = {}
    
    for split in ['train', 'val', 'test']:
        dataset = FaceDataset(
            data_dir=data_dir,
            split=split,
            image_size=image_size,
            augmentation=augmentation and split == 'train'
        )
        
        data_loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == 'train',
            num_workers=num_workers,
            pin_memory=True
        )
    
    return data_loaders
