import numpy as np
from PIL import Image, ImageDraw
from typing import List, Tuple
from .base import BaseAugmentation
from .noise import GaussianNoise
from .geometric import LocalDeformation
from .color import ColorDistortion
import random

class RandomDeletion(BaseAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image)
        height, width = img_np.shape[:2]
        
        num_patches = np.random.randint(1, 4)
        for _ in range(num_patches):
            x = np.random.randint(0, width - width//4)
            y = np.random.randint(0, height - height//4)
            patch_w = np.random.randint(width//8, width//4)
            patch_h = np.random.randint(height//8, height//4)
            img_np[y:y+patch_h, x:x+patch_w] = 0
            
        return Image.fromarray(img_np)

class CenterRedDot(BaseAugmentation):
    def __init__(self, severity: float = 0.7, dot_size: int = 5):
        """
        Add a red dot in the center region of the image.
        
        Args:
            severity: Controls the intensity of red (0.0 to 1.0)
            dot_size: Radius of the dot in pixels
        """
        super().__init__(severity)
        self.dot_size = dot_size
        
    def get_center_region_coords(self, width: int, height: int) -> Tuple[int, int]:
        """Get random coordinates within the center region (middle 3/5 of the image)."""
        # Calculate boundaries for center region
        x_start = width // 5
        x_end = (width * 4) // 5
        y_start = height // 5
        y_end = (height * 4) // 5
        
        # Get random position within center region
        x = np.random.randint(x_start, x_end)
        y = np.random.randint(y_start, y_end)
        
        return x, y
        
    def __call__(self, image: Image.Image) -> Image.Image:
        # Create a copy of the image
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        # Get image dimensions
        width, height = image.size
        
        # Get random position in center region
        x, y = self.get_center_region_coords(width, height)
        
        # Calculate red color intensity based on severity
        red_intensity = int(255 * self.severity)
        
        # Draw the red dot
        bbox = [
            (x - self.dot_size, y - self.dot_size),
            (x + self.dot_size, y + self.dot_size)
        ]
        draw.ellipse(bbox, fill=(red_intensity, 0, 0))
        
        return img_copy

class CenterBlackSquare(BaseAugmentation):
    def __init__(self, severity: float = 0.7, min_size: int = 10, max_size: int = 30):
        """
        Add a black square in the center region of the image.
        
        Args:
            severity: Controls the size of the square relative to min_size and max_size
            min_size: Minimum side length of the square
            max_size: Maximum side length of the square
        """
        super().__init__(severity)
        self.min_size = min_size
        self.max_size = max_size
        
    def get_center_region_coords(self, width: int, height: int, square_size: int) -> Tuple[int, int]:
        """Get random coordinates within the center region (middle 3/5 of the image)."""
        # Calculate boundaries for center region
        x_start = width // 5
        x_end = (width * 4) // 5 - square_size
        y_start = height // 5
        y_end = (height * 4) // 5 - square_size
        
        # Get random position within center region
        x = np.random.randint(x_start, x_end)
        y = np.random.randint(y_start, y_end)
        
        return x, y
        
    def __call__(self, image: Image.Image) -> Image.Image:
        # Create a copy of the image
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        # Get image dimensions
        width, height = image.size
        
        # Calculate square size based on severity
        size_range = self.max_size - self.min_size
        square_size = int(self.min_size + (size_range * self.severity))
        
        # Get random position in center region
        x, y = self.get_center_region_coords(width, height, square_size)
        
        # Draw the black square
        bbox = [
            (x, y),
            (x + square_size, y + square_size)
        ]
        draw.rectangle(bbox, fill=(0, 0, 0))
        
        return img_copy

class AnomalyAugmenter:
    def __init__(self, severity: float = 0.7):
        self.augmentations: List[BaseAugmentation] = [
            GaussianNoise(severity),
            LocalDeformation(severity),
            ColorDistortion(severity),
            RandomDeletion(severity),
            CenterRedDot(severity),
            CenterBlackSquare(severity),
        ]
    
    def generate_anomaly(self, image: Image.Image) -> Image.Image:
        # Generate anomaly images by combining multiple augmentations
        num_augs = np.random.randint(2, 4)
        selected_augs = np.random.choice(self.augmentations, num_augs, replace=False)
        
        img = image
        for aug in selected_augs:
            img = aug(img)
            
        return img