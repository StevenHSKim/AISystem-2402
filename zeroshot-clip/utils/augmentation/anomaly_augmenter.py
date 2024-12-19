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

def get_center_region(width: int, height: int) -> Tuple[int, int, int, int]:
    """
    Get the coordinates of the center 3/5 region of the image.
    
    Args:
        width: Image width
        height: Image height
        
    Returns:
        Tuple[int, int, int, int]: (x_start, y_start, x_end, y_end)
    """
    x_start = width // 5
    x_end = (width * 4) // 5
    y_start = height // 5
    y_end = (height * 4) // 5
    
    return x_start, y_start, x_end, y_end

class Scratch(BaseAugmentation):
    def __init__(self, severity: float = 0.7, num_scratches: Tuple[int, int] = (1, 3),
                 width_range: Tuple[int, int] = (1, 3)):
        """
        Add random scratches to the center region of the image.
        """
        super().__init__(severity)
        self.num_scratches = num_scratches
        self.width_range = width_range
        
    def _generate_scratch_points(self, width: int, height: int) -> List[Tuple[int, int]]:
        """Generate points for a single scratch line within the center region."""
        # Get center region coordinates
        x_start, y_start, x_end, y_end = get_center_region(width, height)
        center_width = x_end - x_start
        center_height = y_end - y_start
        
        # Decide scratch direction (horizontal, vertical, or diagonal)
        direction = random.choice(['h', 'v', 'd'])
        
        if direction == 'h':  # Horizontal scratch
            y = random.randint(y_start, y_end)
            x1 = random.randint(x_start, x_start + center_width // 3)
            x2 = random.randint(x_end - center_width // 3, x_end)
            points = [
                (x1, y),
                (x1 + random.randint(10, 30), y + random.randint(-10, 10)),
                (x2 - random.randint(10, 30), y + random.randint(-10, 10)),
                (x2, y)
            ]
        elif direction == 'v':  # Vertical scratch
            x = random.randint(x_start, x_end)
            y1 = random.randint(y_start, y_start + center_height // 3)
            y2 = random.randint(y_end - center_height // 3, y_end)
            points = [
                (x, y1),
                (x + random.randint(-10, 10), y1 + random.randint(10, 30)),
                (x + random.randint(-10, 10), y2 - random.randint(10, 30)),
                (x, y2)
            ]
        else:  # Diagonal scratch
            x1 = random.randint(x_start, x_start + center_width // 3)
            y1 = random.randint(y_start, y_start + center_height // 3)
            x2 = random.randint(x_end - center_width // 3, x_end)
            y2 = random.randint(y_end - center_height // 3, y_end)
            points = [
                (x1, y1),
                (x1 + random.randint(10, 30), y1 + random.randint(10, 30)),
                (x2 - random.randint(10, 30), y2 - random.randint(10, 30)),
                (x2, y2)
            ]
            
        return points
    
    def __call__(self, image: Image.Image) -> Image.Image:
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        width, height = image.size
        
        # Number of scratches based on severity
        n_scratches = random.randint(
            self.num_scratches[0],
            max(self.num_scratches[0],
                int(self.num_scratches[1] * self.severity))
        )
        
        # Generate scratches
        for _ in range(n_scratches):
            points = self._generate_scratch_points(width, height)
            scratch_width = random.randint(
                self.width_range[0],
                max(self.width_range[0],
                    int(self.width_range[1] * self.severity))
            )
            
            alpha = int(200 * self.severity)
            scratch_color = (255, 255, 255, alpha)
            
            for offset in range(-scratch_width//2, scratch_width//2 + 1):
                offset_points = [(x + random.randint(-1, 1),
                                y + random.randint(-1, 1) + offset)
                               for x, y in points]
                draw.line(offset_points, fill=scratch_color, width=1)
        
        return img_copy

class Crack(BaseAugmentation):
    def __init__(self, severity: float = 0.7, num_cracks: Tuple[int, int] = (1, 2),
                 branching_factor: Tuple[int, int] = (2, 4)):
        """
        Add cracks to the center region of the image.
        """
        super().__init__(severity)
        self.num_cracks = num_cracks
        self.branching_factor = branching_factor
        
    def _generate_crack_segment(self, start: Tuple[int, int], length: int,
                              angle: float, width: int, height: int) -> List[Tuple[int, int]]:
        """Generate a single crack segment within center region bounds."""
        x_start, y_start, x_end, y_end = get_center_region(width, height)
        
        end_x = int(start[0] + length * np.cos(angle))
        end_y = int(start[1] + length * np.sin(angle))
        
        # Add randomness while keeping within center bounds
        end_x = min(max(x_start, end_x + random.randint(-length//4, length//4)), x_end)
        end_y = min(max(y_start, end_y + random.randint(-length//4, length//4)), y_end)
        
        n_points = random.randint(2, 4)
        points = []
        for i in range(n_points + 1):
            t = i / n_points
            rand_x = random.randint(-5, 5) if 0 < i < n_points else 0
            rand_y = random.randint(-5, 5) if 0 < i < n_points else 0
            x = int(start[0] + t * (end_x - start[0])) + rand_x
            y = int(start[1] + t * (end_y - start[1])) + rand_y
            
            # Ensure points stay within center bounds
            x = min(max(x_start, x), x_end)
            y = min(max(y_start, y), y_end)
            points.append((x, y))
            
        return points
    
    def _generate_crack_pattern(self, start: Tuple[int, int], width: int, height: int,
                              depth: int = 0, max_depth: int = 3) -> List[List[Tuple[int, int]]]:
        """Recursively generate crack pattern within center region."""
        if depth >= max_depth:
            return []
            
        patterns = []
        length = random.randint(30, 80) // (depth + 1)
        
        # Generate main segment
        main_angle = random.uniform(0, 2 * np.pi)
        segment = self._generate_crack_segment(start, length, main_angle, width, height)
        patterns.append(segment)
        
        if depth < max_depth:
            n_branches = random.randint(
                self.branching_factor[0],
                max(self.branching_factor[0],
                    int(self.branching_factor[1] * self.severity))
            )
            
            for _ in range(n_branches):
                branch_start_idx = random.randint(0, len(segment) - 1)
                branch_start = segment[branch_start_idx]
                
                branch_angle = main_angle + random.uniform(-np.pi/3, np.pi/3)
                branch_patterns = self._generate_crack_pattern(
                    branch_start, width, height, depth + 1, max_depth
                )
                patterns.extend(branch_patterns)
                
        return patterns
    
    def __call__(self, image: Image.Image) -> Image.Image:
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        width, height = image.size
        
        # Get center region coordinates
        x_start, y_start, x_end, y_end = get_center_region(width, height)
        
        n_cracks = random.randint(
            self.num_cracks[0],
            max(self.num_cracks[0],
                int(self.num_cracks[1] * self.severity))
        )
        
        for _ in range(n_cracks):
            # Random starting point within center region
            start_x = random.randint(x_start, x_end)
            start_y = random.randint(y_start, y_end)
            
            patterns = self._generate_crack_pattern((start_x, start_y), width, height)
            
            for segment in patterns:
                crack_width = max(1, int(2 * self.severity))
                darkness = int(255 * (1 - self.severity * 0.7))
                crack_color = (darkness, darkness, darkness)
                
                draw.line(segment, fill=crack_color, width=crack_width)
                
                for pt in segment:
                    noise_radius = random.randint(0, max(1, int(3 * self.severity)))
                    for _ in range(noise_radius):
                        noise_x = pt[0] + random.randint(-2, 2)
                        noise_y = pt[1] + random.randint(-2, 2)
                        # Ensure noise stays within center region
                        noise_x = min(max(x_start, noise_x), x_end)
                        noise_y = min(max(y_start, noise_y), y_end)
                        draw.point((noise_x, noise_y), fill=crack_color)
        
        return img_copy

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
            Scratch(severity),  # 추가된 Scratch
            Crack(severity)     # 추가된 Crack
        ]
    
    def generate_anomaly(self, image: Image.Image) -> Image.Image:
        # Generate anomaly images by combining multiple augmentations
        num_augs = np.random.randint(2, 4)
        selected_augs = np.random.choice(self.augmentations, num_augs, replace=False)
        
        img = image
        for aug in selected_augs:
            img = aug(img)
            
        return img