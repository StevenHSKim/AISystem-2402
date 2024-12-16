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
        
        num_patches = np.random.randint(1, 3)
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
            dot_size = random.randint(2, 6)
            x = random.randint(0, width)
            y = random.randint(0, height)
            
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
        
        x_center = random.randint(width//4, 3*width//4)
        y_center = random.randint(height//4, 3*height//4)
        
        grid_size = 20
        displacement = int(min(width, height) * 0.15)
        
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
        
        img = image.transform(
            (width, height),
            Image.MESH,
            coords,
            Image.BILINEAR
        )
        
        return img

class GaussianNoise(BaseAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        """가우시안 노이즈 추가"""
        img_np = np.array(image).astype(np.float32)
        noise = np.random.normal(0, self.severity * 25, img_np.shape)  # 감소된 노이즈 강도
        noisy_img = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)

class ColorDistortion(BaseAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        """색상 왜곡"""
        img = image
        
        # 감소된 강도로 색상 변경
        if random.random() < 0.5:
            saturation = ImageEnhance.Color(img)
            img = saturation.enhance(1.0 + (self.severity - 0.5) * 0.5)
        
        if random.random() < 0.5:
            brightness = ImageEnhance.Brightness(img)
            img = brightness.enhance(1.0 + (self.severity - 0.5) * 0.3)
        
        if random.random() < 0.5:
            contrast = ImageEnhance.Contrast(img)
            img = contrast.enhance(1.0 + (self.severity - 0.5) * 0.3)
        
        return img

class LocalDeformation(BaseAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        """부분적 변형"""
        width, height = image.size
        x1 = np.random.randint(0, width // 2)
        y1 = np.random.randint(0, height // 2)
        x2 = x1 + width // 4
        y2 = y1 + height // 4
        
        img_np = np.array(image)
        region = img_np[y1:y2, x1:x2]
        distorted = np.roll(region, shift=int(self.severity * 10))  # 감소된 변형 강도
        img_np[y1:y2, x1:x2] = distorted
        
        return Image.fromarray(img_np)

class AnomalyAugmenter:
    def __init__(self, severity: float = 0.7):
        self.severity = severity
        self.primary_augmentations = [
            (RandomDeletion(severity), 0.4),
            (RedDotAnomaly(severity), 0.3),
            (DeformationAnomaly(severity), 0.3)
        ]
        self.secondary_augmentations = [
            (GaussianNoise(severity * 0.5), 0.4),
            (ColorDistortion(severity * 0.3), 0.3),
            (LocalDeformation(severity * 0.4), 0.3)
        ]
    
    def generate_anomaly(self, image: Image.Image) -> Image.Image:
        """여러 augmentation을 조합하여 anomaly 생성"""
        try:
            img = image.copy()
            
            # Primary augmentation 적용 (1-2개)
            num_primary = random.randint(1, 2)
            selected_primary = random.choices(
                [aug for aug, _ in self.primary_augmentations],
                weights=[w for _, w in self.primary_augmentations],
                k=num_primary
            )
            
            # Primary augmentation 적용
            for aug in selected_primary:
                try:
                    img = aug(img)
                except Exception as e:
                    print(f"Error in primary augmentation: {str(e)}")
                    continue
            
            # Secondary augmentation 적용 (70% 확률)
            if random.random() < 0.7:
                aug, _ = random.choices(
                    self.secondary_augmentations,
                    weights=[w for _, w in self.secondary_augmentations],
                    k=1
                )[0]
                try:
                    img = aug(img)
                except Exception as e:
                    print(f"Error in secondary augmentation: {str(e)}")
            
            # mild blur 적용 (30% 확률)
            img = self._apply_mild_blur(img)
            
            return img
            
        except Exception as e:
            print(f"Error in generate_anomaly: {str(e)}")
            return image  # 에러 발생 시 원본 이미지 반환
    
    def _apply_mild_blur(self, image: Image.Image) -> Image.Image:
        """약한 블러 효과 적용 (선택적)"""
        if random.random() < 0.3:  # 30% 확률로 적용
            return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.0)))
        return image