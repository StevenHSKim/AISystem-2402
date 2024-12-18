import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import cv2
import random
from typing import List
from .base import BaseAugmentation
from typing import Optional, Tuple

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

class RandomDeletion(BaseAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        try:
            img_np = np.array(image)
            height, width = img_np.shape[:2]
            
            if height < 4 or width < 4:
                return image
            
            rng = np.random.RandomState(42)
            
            # 더 다양한 크기의 삭제 영역 생성
            div_factor = rng.randint(4, 8)  # 4에서 8 사이의 랜덤 값
            patch_w = max(width // div_factor, 1)
            patch_h = max(height // div_factor, 1)
            
            x = rng.randint(0, max(1, width - patch_w))
            y = rng.randint(0, max(1, height - patch_h))
            
            y_start = max(0, y-3)
            y_end = min(height, y+patch_h+3)
            x_start = max(0, x-3)
            x_end = min(width, x+patch_w+3)
            
            if y_end > y_start and x_end > x_start:
                surrounding = img_np[y_start:y_end, x_start:x_end]
                if surrounding.size > 0:
                    fill_color = np.mean(surrounding.reshape(-1, surrounding.shape[-1]), axis=0)
                    
                    noise = rng.normal(0, 2, (patch_h, patch_w, 3))
                    fill_area = np.tile(fill_color, (patch_h, patch_w, 1))
                    fill_area = np.clip(fill_area + noise, 0, 255)
                    
                    img_np[y:min(y+patch_h, height), x:min(x+patch_w, width)] = fill_area
            
            return Image.fromarray(img_np.astype(np.uint8))
            
        except Exception as e:
            print(f"Error in RandomDeletion: {str(e)}")
            return image

class RedDotAnomaly(BaseAugmentation):
    def __init__(self, severity: float):
        super().__init__(severity)
        self.rng = random.Random(42)  # 클래스별 random generator
        
    def __call__(self, image: Image.Image) -> Image.Image:
        """빨간 점 추가"""
        try:
            img = image.copy()
            draw = ImageDraw.Draw(img, 'RGBA')
            width, height = img.size
            
            if width < 4 or height < 4:
                return image
            
            num_dots = self.rng.randint(1, 3)
            for _ in range(num_dots):
                dot_size = self.rng.randint(2, min(6, min(width, height) // 4))
                x = self.rng.randint(dot_size, max(dot_size + 1, width - dot_size))
                y = self.rng.randint(dot_size, max(dot_size + 1, height - dot_size))
                
                red_color = (255, 0, 0, self.rng.randint(150, 255))
                draw.ellipse(
                    [(x-dot_size, y-dot_size), (x+dot_size, y+dot_size)],
                    fill=red_color
                )
            
            return img
            
        except Exception as e:
            print(f"Error in RedDotAnomaly: {str(e)}")
            return image        

class LocalDeformation(BaseAugmentation):
    def __init__(self, severity: float):
        super().__init__(severity)
        self.rng = random.Random(42)  # 클래스별 random generator
        
    def __call__(self, image: Image.Image) -> Image.Image:
        """부분적 변형"""
        try:
            img_np = np.array(image)
            height, width = img_np.shape[:2]
            
            if height < 4 or width < 4:
                return image
            
            region_w = max(width // 4, 1)
            region_h = max(height // 4, 1)
            x1 = self.rng.randint(0, max(1, width - region_w))
            y1 = self.rng.randint(0, max(1, height - region_h))
            
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
    def __init__(self, severity: float):
        super().__init__(severity)
        self.rng = np.random.RandomState(42)  # 클래스별 NumPy random generator
        
    def __call__(self, image: Image.Image) -> Image.Image:
        """가우시안 노이즈 추가"""
        try:
            img_np = np.array(image).astype(np.float32)
            
            noise_level = max(1e-3, self.severity * 25.0)
            noise = self.rng.normal(0, noise_level, img_np.shape)
            
            noisy_img = img_np + noise
            noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
            
            return Image.fromarray(noisy_img)
        
        except Exception as e:
            print(f"Error in GaussianNoise: {str(e)}")
            return image


class AnomalyAugmenter:
    def __init__(self, severity: float = 0.5):
        self.severity = severity
        self.rng = np.random.RandomState(42)
        
        # 증강 기법들의 가중치 설정
        self.augmentation_weights = {
            'pattern_injection': 0.3,
            'texture_corruption': 0.25,
            'random_cuts': 0.25,
            'deformation': 0.2
        }
        
    def generate_anomaly(self, image: Image.Image) -> Optional[Image.Image]:
        try:
            img = image.copy()
            
            # 여러 증강 기법 중 하나를 랜덤하게 선택
            aug_type = self.rng.choice(
                list(self.augmentation_weights.keys()),
                p=list(self.augmentation_weights.values())
            )
            
            if aug_type == 'pattern_injection':
                img = self._apply_pattern_injection(img)
            elif aug_type == 'texture_corruption':
                img = self._apply_texture_corruption(img)
            elif aug_type == 'random_cuts':
                img = self._apply_random_cuts(img)
            else:  # deformation
                img = self._apply_deformation(img)
                
            return img
            
        except Exception as e:
            print(f"Error in anomaly generation: {str(e)}")
            return image
            
    def _apply_pattern_injection(self, image: Image.Image) -> Image.Image:
        """다양한 패턴(점, 선, 얼룩 등) 삽입"""
        try:
            img = image.copy()
            draw = ImageDraw.Draw(img, 'RGBA')
            width, height = img.size
            
            # 패턴 유형 랜덤 선택
            pattern_type = self.rng.choice(['dots', 'scratches', 'stains'])
            
            if pattern_type == 'dots':
                # 다중 점 생성
                num_dots = self.rng.randint(1, 4)
                for _ in range(num_dots):
                    # 크기, 색상, 투명도 다양화
                    size = self.rng.randint(2, max(3, min(width, height) // 20))
                    color = (
                        self.rng.randint(200, 255),  # R
                        self.rng.randint(0, 50),     # G
                        self.rng.randint(0, 50),     # B
                        self.rng.randint(150, 255)   # A
                    )
                    x = self.rng.randint(size, width - size)
                    y = self.rng.randint(size, height - size)
                    draw.ellipse([(x-size, y-size), (x+size, y+size)], fill=color)
                    
            elif pattern_type == 'scratches':
                # 긁힘 효과
                num_scratches = self.rng.randint(1, 3)
                for _ in range(num_scratches):
                    start_x = self.rng.randint(0, width)
                    start_y = self.rng.randint(0, height)
                    end_x = start_x + self.rng.randint(-width//4, width//4)
                    end_y = start_y + self.rng.randint(-height//4, height//4)
                    draw.line([(start_x, start_y), (end_x, end_y)],
                            fill=(150, 150, 150, 200),
                            width=self.rng.randint(1, 3))
                    
            else:  # stains
                # 얼룩 효과
                num_stains = self.rng.randint(1, 2)
                for _ in range(num_stains):
                    x = self.rng.randint(0, width)
                    y = self.rng.randint(0, height)
                    size = self.rng.randint(5, max(6, min(width, height) // 10))
                    color = (
                        self.rng.randint(100, 200),
                        self.rng.randint(100, 200),
                        self.rng.randint(100, 200),
                        self.rng.randint(100, 180)
                    )
                    draw.ellipse([(x-size, y-size), (x+size, y+size)], fill=color)
                    
            return img
            
        except Exception as e:
            print(f"Error in pattern injection: {str(e)}")
            return image
            
    def _apply_texture_corruption(self, image: Image.Image) -> Image.Image:
        """텍스처 변형(블러, 노이즈 등)"""
        try:
            img = np.array(image)
            
            # 변형 유형 선택
            corruption_type = self.rng.choice(['blur', 'noise', 'salt_pepper'])
            
            if corruption_type == 'blur':
                # 가우시안 블러
                kernel_size = self.rng.choice([3, 5, 7])
                img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
                
            elif corruption_type == 'noise':
                # 가우시안 노이즈
                noise = self.rng.normal(0, 25, img.shape)
                img = np.clip(img + noise, 0, 255).astype(np.uint8)
                
            else:  # salt_pepper
                # Salt & Pepper 노이즈
                prob = self.rng.uniform(0.01, 0.05)
                thresh = 1 - prob
                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        rdn = self.rng.rand()
                        if rdn < prob:
                            img[i][j] = 0
                        elif rdn > thresh:
                            img[i][j] = 255
                            
            return Image.fromarray(img)
            
        except Exception as e:
            print(f"Error in texture corruption: {str(e)}")
            return image
            
    def _apply_random_cuts(self, image: Image.Image) -> Image.Image:
        """랜덤한 영역 절단 또는 제거"""
        try:
            img_np = np.array(image)
            height, width = img_np.shape[:2]
            
            # 절단 영역 크기 및 위치 설정
            cut_type = self.rng.choice(['rectangle', 'triangle'])
            
            if cut_type == 'rectangle':
                # 직사각형 절단
                cut_width = self.rng.randint(width // 8, width // 4)
                cut_height = self.rng.randint(height // 8, height // 4)
                x = self.rng.randint(0, width - cut_width)
                y = self.rng.randint(0, height - cut_height)
                
                # 배경색으로 채우기
                fill_color = img_np[max(0, y-5):min(height, y+5), max(0, x-5):min(width, x+5)].mean(axis=(0, 1))
                img_np[y:y+cut_height, x:x+cut_width] = fill_color
                
            else:  # triangle
                # 삼각형 절단
                points = np.array([
                    [self.rng.randint(0, width), self.rng.randint(0, height)],
                    [self.rng.randint(0, width), self.rng.randint(0, height)],
                    [self.rng.randint(0, width), self.rng.randint(0, height)]
                ])
                
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(mask, [points], 255)
                
                # 배경색으로 채우기
                fill_color = img_np[mask > 0].mean(axis=0)
                img_np[mask > 0] = fill_color
                
            return Image.fromarray(img_np)
            
        except Exception as e:
            print(f"Error in random cuts: {str(e)}")
            return image
            
    def _apply_deformation(self, image: Image.Image) -> Image.Image:
        """지역적 변형 적용"""
        try:
            img_np = np.array(image)
            height, width = img_np.shape[:2]
            
            # 변형 강도 설정
            strength = self.rng.uniform(5, 15)
            
            # 변형 영역 설정
            region_size = self.rng.randint(
                min(width, height) // 6,
                min(width, height) // 3
            )
            
            # 변형 중심점
            center_x = self.rng.randint(region_size, width - region_size)
            center_y = self.rng.randint(region_size, height - region_size)
            
            # 변형 맵 생성
            y, x = np.mgrid[:height, :width]
            dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # 변형 마스크 생성
            mask = dist_from_center < region_size
            
            # 변형이 적용될 픽셀 위치 계산
            angle = self.rng.uniform(0, 2*np.pi)
            displacement = np.maximum(0, (1 - dist_from_center/region_size)) * strength
            dx = np.cos(angle) * displacement
            dy = np.sin(angle) * displacement
            
            # 새로운 좌표 계산
            x_new = x.astype(np.float32)
            y_new = y.astype(np.float32)
            x_new[mask] += dx[mask]
            y_new[mask] += dy[mask]
            
            # 범위 제한
            x_new = np.clip(x_new, 0, width-1)
            y_new = np.clip(y_new, 0, height-1)
            
            # 보간법을 사용한 이미지 변형
            from scipy.ndimage import map_coordinates
            
            deformed = np.zeros_like(img_np)
            for channel in range(img_np.shape[2]):
                deformed[..., channel] = map_coordinates(
                    img_np[..., channel],
                    [y_new, x_new],
                    order=1,
                    mode='reflect'
                )
                
            return Image.fromarray(deformed.astype(np.uint8))
            
        except Exception as e:
            print(f"Error in deformation: {str(e)}")
            return image