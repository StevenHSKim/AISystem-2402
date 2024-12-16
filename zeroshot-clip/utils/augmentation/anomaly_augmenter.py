import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import cv2
import random
from typing import List
from .base import BaseAugmentation

class RandomDeletion(BaseAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        """더 자연스러운 부품 누락 효과"""
        img_np = np.array(image)
        height, width = img_np.shape[:2]
        
        num_patches = np.random.randint(1, 3)  # 좀 더 적은 수의 패치
        for _ in range(num_patches):
            x = np.random.randint(0, width - width//4)
            y = np.random.randint(0, height - height//4)
            patch_w = np.random.randint(width//8, width//4)
            patch_h = np.random.randint(height//8, height//4)
            
            # 주변 픽셀 색상으로 채우기
            surrounding = img_np[
                max(0, y-2):min(height, y+patch_h+2),
                max(0, x-2):min(width, x+patch_w+2)
            ]
            fill_color = np.median(surrounding, axis=(0, 1))
            img_np[y:y+patch_h, x:x+patch_w] = fill_color
            
        return Image.fromarray(img_np)

class RedDotAnomaly(BaseAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        """빨간 점 추가"""
        img = image.copy()
        draw = ImageDraw.Draw(img, 'RGBA')
        width, height = img.size
        
        num_dots = random.randint(1, 3)
        for _ in range(num_dots):
            # 점 크기와 위치
            dot_size = random.randint(2, 6)
            x = random.randint(0, width)
            y = random.randint(0, height)
            
            # 빨간색 점 (약간의 투명도)
            red_color = (255, 0, 0, random.randint(150, 255))
            draw.ellipse(
                [x-dot_size, y-dot_size, x+dot_size, y+dot_size],
                fill=red_color
            )
        
        return img

class DeformationAnomaly(BaseAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        """찌그러짐 효과"""
        width, height = image.size
        
        # 변형 지점 선택
        x_center = random.randint(width//4, 3*width//4)
        y_center = random.randint(height//4, 3*height//4)
        
        # 격자 생성
        grid_size = 20
        displacement = int(min(width, height) * 0.15)
        
        # 변형 매트릭스 생성
        coords = []
        for y in range(0, height, grid_size):
            for x in range(0, width, grid_size):
                dx = x - x_center
                dy = y - y_center
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance < displacement * 3:
                    factor = (1 - distance/(displacement * 3)) * displacement
                    new_x = x + int(dx * factor / distance) if distance > 0 else x
                    new_y = y + int(dy * factor / distance) if distance > 0 else y
                else:
                    new_x, new_y = x, y
                    
                coords.extend([x, y, new_x, new_y])
        
        # 변형 적용
        img = image.transform(
            (width, height),
            Image.MESH,
            coords,
            Image.BILINEAR
        )
        
        return img

class AnomalyAugmenter:
    def __init__(self, severity: float = 0.7):
        self.severity = severity
        self.augmentations: List[BaseAugmentation] = [
            RandomDeletion(severity),    # 부품 누락
            RedDotAnomaly(severity),     # 빨간 점
            DeformationAnomaly(severity) # 찌그러짐
        ]
    
    def generate_anomaly(self, image: Image.Image) -> Image.Image:
        # 각 augmentation 유형별 가중치 설정
        weights = [0.4, 0.3, 0.3]  # 부품 누락, 빨간 점, 찌그러짐 순
        
        # 1-2개의 augmentation 랜덤 선택
        num_augs = random.randint(1, 2)
        selected_augs = random.choices(
            self.augmentations,
            weights=weights,
            k=num_augs
        )
        
        img = image
        for aug in selected_augs:
            img = aug(img)
            
        return img

    def _apply_mild_noise(self, image: Image.Image) -> Image.Image:
        """약간의 노이즈 추가 (선택적)"""
        if random.random() < 0.3:  # 30% 확률로 적용
            img_np = np.array(image)
            noise = np.random.normal(0, 10, img_np.shape)  # 약한 노이즈
            img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
            return Image.fromarray(img_np)
        return image