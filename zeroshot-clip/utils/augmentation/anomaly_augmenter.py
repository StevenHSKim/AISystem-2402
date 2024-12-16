import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import cv2
import random
from typing import List
from .base import BaseAugmentation

class RandomDeletion(BaseAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        """더 자연스러운 부품 누락 효과"""
        try:
            img_np = np.array(image)
            height, width = img_np.shape[:2]
            
            if height < 4 or width < 4:  # 너무 작은 이미지는 처리하지 않음
                return image
            
            num_patches = np.random.randint(1, 3)
            for _ in range(num_patches):
                patch_w = max(width//8, 1)
                patch_h = max(height//8, 1)
                x = np.random.randint(0, max(1, width - patch_w))
                y = np.random.randint(0, max(1, height - patch_h))
                
                y_start = max(0, y-2)
                y_end = min(height, y+patch_h+2)
                x_start = max(0, x-2)
                x_end = min(width, x+patch_w+2)
                
                if y_end > y_start and x_end > x_start:  # 유효한 영역인지 확인
                    surrounding = img_np[y_start:y_end, x_start:x_end]
                    if surrounding.size > 0:
                        fill_color = np.median(surrounding.reshape(-1, surrounding.shape[-1]), axis=0)
                        img_np[y:min(y+patch_h, height), x:min(x+patch_w, width)] = fill_color
            
            return Image.fromarray(img_np)
            
        except Exception as e:
            print(f"Error in RandomDeletion: {str(e)}")
            return image

class RedDotAnomaly(BaseAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        """빨간 점 추가"""
        try:
            img = image.copy()
            draw = ImageDraw.Draw(img, 'RGBA')
            width, height = img.size
            
            if width < 4 or height < 4:  # 너무 작은 이미지는 처리하지 않음
                return image
            
            num_dots = random.randint(1, 3)
            for _ in range(num_dots):
                dot_size = random.randint(2, min(6, min(width, height) // 4))
                x = random.randint(dot_size, max(dot_size + 1, width - dot_size))
                y = random.randint(dot_size, max(dot_size + 1, height - dot_size))
                
                red_color = (255, 0, 0, random.randint(150, 255))
                draw.ellipse(
                    [(x-dot_size, y-dot_size), (x+dot_size, y+dot_size)],
                    fill=red_color
                )
            
            return img
            
        except Exception as e:
            print(f"Error in RedDotAnomaly: {str(e)}")
            return image


class DeformationAnomaly(BaseAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        """찌그러짐 효과 - OpenCV 기반 변형"""
        try:
            # PIL Image를 numpy array로 변환
            img_np = np.array(image)
            
            # 이미지가 너무 작으면 변형하지 않음
            height, width = img_np.shape[:2]
            if width < 32 or height < 32:
                return image
            
            # 변형 맵 생성
            map_x = np.zeros((height, width), dtype=np.float32)
            map_y = np.zeros((height, width), dtype=np.float32)
            
            # 중심점
            center_x, center_y = width // 2, height // 2
            max_dist = np.sqrt(center_x**2 + center_y**2)
            
            # 각 픽셀에 대한 변형 계산
            for y in range(height):
                for x in range(width):
                    # 중심으로부터의 거리 계산
                    dx = x - center_x
                    dy = y - center_y
                    dist = np.sqrt(dx**2 + dy**2) + 1e-6  # 0으로 나누기 방지
                    
                    # 변형 강도 계산 (중심에서 멀어질수록 감소)
                    strength = max(0.1, 1.0 - (dist / max_dist))
                    
                    # 변형량 계산
                    offset_x = dx * strength * self.severity * 0.1
                    offset_y = dy * strength * self.severity * 0.1
                    
                    # 새로운 좌표 계산
                    new_x = x + offset_x
                    new_y = y + offset_y
                    
                    # 경계 확인
                    map_x[y, x] = np.clip(new_x, 0, width - 1)
                    map_y[y, x] = np.clip(new_y, 0, height - 1)
            
            # 변형 적용
            deformed = cv2.remap(img_np, map_x, map_y, 
                               interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
            
            # numpy array를 PIL Image로 변환
            return Image.fromarray(deformed)
            
        except Exception as e:
            print(f"Error in DeformationAnomaly: {str(e)}")
            return image
        

class ColorDistortion(BaseAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        """색상 왜곡"""
        try:
            img = image.copy()
            
            params = {
                'Color': random.uniform(0.8, 1.2),
                'Brightness': random.uniform(0.8, 1.2),
                'Contrast': random.uniform(0.8, 1.2)
            }
            
            for enhancer_type, factor in params.items():
                try:
                    if random.random() < 0.5:
                        enhancer = getattr(ImageEnhance, enhancer_type)(img)
                        img = enhancer.enhance(factor)
                except Exception as e:
                    print(f"Error in {enhancer_type} enhancement: {str(e)}")
                    continue
            
            return img
            
        except Exception as e:
            print(f"Error in ColorDistortion: {str(e)}")
            return image

class LocalDeformation(BaseAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        """부분적 변형"""
        try:
            img_np = np.array(image)
            height, width = img_np.shape[:2]
            
            if height < 4 or width < 4:  # 너무 작은 이미지는 처리하지 않음
                return image
            
            region_w = max(width // 4, 1)
            region_h = max(height // 4, 1)
            x1 = random.randint(0, max(1, width - region_w))
            y1 = random.randint(0, max(1, height - region_h))
            
            region = img_np[y1:y1+region_h, x1:x1+region_w].copy()
            if region.size > 0:
                try:
                    shift_x = max(1, int(self.severity * region_w * 0.1))
                    shift_y = max(1, int(self.severity * region_h * 0.1))
                    
                    region = np.roll(region, shift=(shift_y, shift_x), axis=(0, 1))
                    img_np[y1:y1+region_h, x1:x1+region_w] = region
                    
                except Exception as e:
                    print(f"Error in region deformation: {str(e)}")
            
            return Image.fromarray(img_np)
            
        except Exception as e:
            print(f"Error in LocalDeformation: {str(e)}")
            return image

class GaussianNoise(BaseAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        """가우시안 노이즈 추가"""
        try:
            img_np = np.array(image).astype(np.float32)
            
            # noise_level이 너무 작지 않도록 조정
            noise_level = max(1e-3, self.severity * 25.0)
            noise = np.random.normal(0, noise_level, img_np.shape)
            
            noisy_img = img_np + noise
            noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
            
            return Image.fromarray(noisy_img)
        
        except Exception as e:
            print(f"Error in GaussianNoise: {str(e)}")
            return image

class AnomalyAugmenter:
    def __init__(self, severity: float = 0.7):
        self.severity = max(0.1, min(0.3, severity))  # severity 범위를 더 제한
        
        # Deformation의 severity를 더욱 낮춤
        self.primary_augmentations = [
            RandomDeletion(self.severity),
            RedDotAnomaly(self.severity),
            DeformationAnomaly(self.severity * 0.2)  # severity를 20%로 제한
        ]
        # Deformation의 선택 확률을 낮춤
        self.primary_weights = [0.45, 0.45, 0.1]
        
        self.secondary_augmentations = [
            GaussianNoise(self.severity * 0.5),
            ColorDistortion(self.severity * 0.3),
            LocalDeformation(self.severity * 0.3)
        ]
        self.secondary_weights = [0.4, 0.4, 0.2]
    
    def generate_anomaly(self, image: Image.Image) -> Image.Image:
        try:
            if not isinstance(image, Image.Image):
                raise ValueError("Input must be a PIL Image")
            
            if image.size[0] < 64 or image.size[1] < 64:
                return image
                
            img = image.copy()
            
            # Primary augmentation 적용 (DeformationAnomaly 사용 확률 제한)
            if len(self.primary_augmentations) > 0:
                aug_idx = random.choices(
                    population=range(len(self.primary_augmentations)),
                    weights=self.primary_weights,
                    k=1
                )[0]
                
                try:
                    # Deformation인 경우 추가 검증
                    if isinstance(self.primary_augmentations[aug_idx], DeformationAnomaly):
                        if random.random() < 0.3:  # 30% 확률로만 적용
                            img = self.primary_augmentations[aug_idx](img)
                    else:
                        img = self.primary_augmentations[aug_idx](img)
                except Exception as e:
                    print(f"Primary augmentation failed: {str(e)}")
            
            # Secondary augmentation (확률 낮춤)
            if random.random() < 0.3:
                try:
                    aug_idx = random.choices(
                        population=range(len(self.secondary_augmentations)),
                        weights=self.secondary_weights,
                        k=1
                    )[0]
                    img = self.secondary_augmentations[aug_idx](img)
                except Exception as e:
                    print(f"Secondary augmentation failed: {str(e)}")
            
            # Blur는 낮은 확률로만 적용
            if random.random() < 0.2:
                try:
                    radius = random.uniform(0.2, 0.5)
                    img = img.filter(ImageFilter.GaussianBlur(radius=radius))
                except Exception:
                    pass
            
            return img
            
        except Exception as e:
            print(f"Error in generate_anomaly: {str(e)}")
            return image