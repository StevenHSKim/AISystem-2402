import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
from utils.augmentation.anomaly_augmenter import AnomalyAugmenter
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MiniBatchKMeans
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)

class ContrastiveMemoryBank:
    def __init__(self, feature_dim: int, device: str, memory_size: int = 300):
        """
        향상된 Contrastive Memory Bank 초기화
        
        Args:
            feature_dim: 특징 벡터의 차원
            device: 사용할 디바이스 (cuda/cpu/mps)
            memory_size: 메모리 뱅크의 크기
        """
        self.memory_size = memory_size
        self.temperature = 0.07  # temperature 미세 조정
        self.features = None
        self.ptr = 0
        self.device = device
        self.feature_dim = feature_dim
        
        # 통계 추적을 위한 변수들
        self.mean = None
        self.std = None
        self.running_max = None
        
    def update(self, features: torch.Tensor):
        """
        메모리 뱅크 업데이트
        
        Args:
            features: 새로운 특징 벡터들 (batch_size x feature_dim)
        """
        batch_size = features.size(0)
        
        # 초기 메모리 설정
        if self.features is None:
            torch.manual_seed(42)
            self.features = torch.randn(self.memory_size, features.size(1)).to(self.device)
            self.features = self._normalize_features(self.features)
            
            # 통계 초기화
            self.mean = torch.mean(self.features, dim=0)
            self.std = torch.std(self.features, dim=0)
            self.running_max = torch.max(torch.abs(self.features))
        
        # 포인터 순환
        if self.ptr + batch_size >= self.memory_size:
            self.ptr = 0
        
        # 특징 정규화
        normalized_features = self._normalize_features(features.to(self.device))
        
        # 메모리 업데이트
        self.features[self.ptr:self.ptr + batch_size] = normalized_features
        
        # 통계 업데이트
        self._update_statistics()
        
        # 포인터 업데이트
        self.ptr = (self.ptr + batch_size) % self.memory_size
        
    def get_similarity(self, query: torch.Tensor) -> torch.Tensor:
        if self.features is None:
            return torch.zeros(query.size(0)).to(self.device)
        
        # 쿼리 정규화
        query = self._normalize_features(query.to(self.device))
        
        # 유사도 계산
        sim = self._compute_similarity(query)
        
        # Top-k 유사도 평균
        k = min(5, self.features.size(0))  # k 증가
        top_k_sim = torch.topk(sim, k=k, dim=1)[0]
        
        # 이상치 제거를 위한 weighted mean
        weights = torch.softmax(-torch.arange(k).float() / 2, dim=0).to(self.device)
        weighted_mean = (top_k_sim * weights.unsqueeze(0)).sum(dim=1)
        
        return weighted_mean / self.temperature
    
    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        특징 벡터 정규화
        """
        # L2 정규화
        features = F.normalize(features, dim=1, p=2)
        
        # 추가 정규화
        if self.running_max is not None:
            features = features / self.running_max
            
        return features
    
    def _compute_similarity(self, query: torch.Tensor) -> torch.Tensor:
        """
        개선된 유사도 계산
        """
        # 코사인 유사도
        cos_sim = F.cosine_similarity(
            query.unsqueeze(1),
            self.features.unsqueeze(0),
            dim=2
        )
        
        # L2 거리
        l2_dist = torch.cdist(query, self.features)
        l2_sim = 1 / (1 + l2_dist)
        
        # 결합된 유사도
        combined_sim = (cos_sim + l2_sim) / 2
        
        return combined_sim
    
    def _update_statistics(self):
        """
        메모리 뱅크 통계 업데이트
        """
        with torch.no_grad():
            # 이동 평균 업데이트
            current_mean = torch.mean(self.features, dim=0)
            current_std = torch.std(self.features, dim=0)
            current_max = torch.max(torch.abs(self.features))
            
            momentum = 0.9
            self.mean = momentum * self.mean + (1 - momentum) * current_mean
            self.std = momentum * self.std + (1 - momentum) * current_std
            self.running_max = max(self.running_max * momentum, current_max)
  
class AnomalyDetector:
    def __init__(self, model):
        self.model = model
        self.class_thresholds = {}  # 클래스별 threshold를 저장할 딕셔너리
        self.class_embeddings = None
        self.anomaly_embeddings = None
        self.memory_bank = None
        
    def prepare(self, normal_samples: Dict[str, List[str]]) -> None:
        print("Computing class embeddings...")
        self.class_embeddings = self._compute_class_embeddings(normal_samples)
        
        print("Generating anomaly embeddings...")
        self.anomaly_embeddings = self._generate_anomaly_embeddings(normal_samples)
        
        print("Initializing memory bank...")
        feature_dim = next(iter(self.class_embeddings.values())).shape[-1]
        self.memory_bank = ContrastiveMemoryBank(feature_dim=feature_dim, device=self.model.device)
        self._initialize_memory_bank(normal_samples)
        
        print("Finding optimal thresholds per class...")
        validation_scores = self._generate_validation_scores_per_class(normal_samples)
        self.class_thresholds = self._find_optimal_thresholds_per_class(validation_scores)
        for class_name, threshold in self.class_thresholds.items():
            print(f"Optimal threshold for {class_name}: {threshold:.3f}")

    def predict(self, image: torch.Tensor) -> Dict:
        try:
            features = self.model.extract_features(image)
            
            if features is None:
                raise ValueError("Failed to extract features from image")
            
            # 모든 클래스와의 유사도 계산
            class_similarities = {}
            for class_name, class_embedding in self.class_embeddings.items():
                class_embedding = F.normalize(class_embedding.to(self.model.device), dim=-1)
                similarity = F.cosine_similarity(features, class_embedding.unsqueeze(0))
                class_similarities[class_name] = similarity.item()
            
            # 가장 유사한 클래스 선택
            most_similar_class = max(class_similarities.items(), key=lambda x: x[1])[0]
            threshold = self.class_thresholds[most_similar_class]
            
            # 해당 클래스의 threshold로 점수 계산
            score, normal_sim, anomaly_sim = self._compute_anomaly_score(features)
            is_anomaly = score > threshold
            
            return {
                'predicted_label': 'anomaly' if is_anomaly else 'normal',
                'anomaly_score': float(score),
                'normal_similarity': float(normal_sim),
                'anomaly_similarity': float(anomaly_sim),
                'is_anomaly': bool(is_anomaly),
                'threshold': float(threshold),
                'most_similar_class': most_similar_class
            }
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return {
                'predicted_label': 'error',
                'anomaly_score': 1.0,
                'normal_similarity': 0.0,
                'anomaly_similarity': 1.0,
                'is_anomaly': True,
                'threshold': 0.5,
                'most_similar_class': None
            }
            
    def _compute_class_embeddings(self, samples_dict: Dict[str, List[str]]) -> Dict[str, torch.Tensor]:
        class_embeddings = {}
        
        for class_name, image_paths in tqdm(samples_dict.items(), desc="Computing class embeddings"):
            embeddings = []
            # 각 이미지에 대한 여러 각도의 feature 추출
            for img_path in image_paths:
                try:
                    image = Image.open(img_path).convert('RGB')
                    # 원본 이미지의 feature
                    image_input = self.model.preprocess(image).unsqueeze(0).to(self.model.device)
                    features = self.model.extract_features(image_input)
                    embeddings.append(features)
                    
                    # 약간의 회전과 크기 변경을 통한 추가 feature
                    for angle in [-10, 10]:  # 회전 각도
                        rotated = image.rotate(angle, Image.BILINEAR)
                        input_tensor = self.model.preprocess(rotated).unsqueeze(0).to(self.model.device)
                        rot_features = self.model.extract_features(input_tensor)
                        embeddings.append(rot_features)
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
                    continue
            
            if embeddings:
                stacked_embeddings = torch.cat(embeddings, dim=0)
                
                # Outlier 제거
                mean_embedding = torch.mean(stacked_embeddings, dim=0)
                distances = torch.norm(stacked_embeddings - mean_embedding, dim=1)
                threshold = torch.mean(distances) + torch.std(distances)
                mask = distances < threshold
                filtered_embeddings = stacked_embeddings[mask]
                
                # 최종 class embedding 계산
                class_embedding = torch.mean(filtered_embeddings, dim=0)
                class_embeddings[class_name] = F.normalize(class_embedding, dim=-1)
        
        return class_embeddings
    
    def _initialize_memory_bank(self, samples_dict: Dict[str, List[str]]) -> None:
        class_features = []
        
        for class_name, image_paths in samples_dict.items():
            for img_path in image_paths:
                try:
                    image = Image.open(img_path).convert('RGB')
                    image_input = self.model.preprocess(image).unsqueeze(0).to(self.model.device)
                    features = self.model.extract_features(image_input)
                    class_features.append(features)
                    
                    # 더 다양한 normal augmentation 적용
                    for _ in range(3):  # augmentation 횟수 증가
                        aug_image = self._apply_normal_augmentation(image)
                        aug_input = self.model.preprocess(aug_image).unsqueeze(0).to(self.model.device)
                        aug_features = self.model.extract_features(aug_input)
                        class_features.append(aug_features)
                        
                except Exception as e:
                    print(f"Error in memory bank initialization: {str(e)}")
                    continue
        
        if class_features:
            stacked_features = torch.cat(class_features, dim=0)
            # Feature clustering을 통한 대표 샘플 선택
            n_clusters = min(len(stacked_features), self.memory_bank.memory_size)
            indices = torch.randperm(len(stacked_features))[:n_clusters]
            selected_features = stacked_features[indices]
            self.memory_bank.update(F.normalize(selected_features, dim=-1))

    def _apply_normal_augmentation(self, image: Image.Image) -> Image.Image:
        try:
            img = image.copy()
            
            # Brightness
            if random.random() < 0.8:
                enhancer = ImageEnhance.Brightness(img)
                factor = random.uniform(0.7, 1.3)
                img = enhancer.enhance(factor)
            
            # Contrast
            if random.random() < 0.8:
                enhancer = ImageEnhance.Contrast(img)
                factor = random.uniform(0.7, 1.3)
                img = enhancer.enhance(factor)
            
            # Color
            if random.random() < 0.5:
                enhancer = ImageEnhance.Color(img)
                factor = random.uniform(0.8, 1.2)
                img = enhancer.enhance(factor)
            
            # Random rotation
            if random.random() < 0.6:
                angle = random.uniform(-15, 15)
                img = img.rotate(angle, Image.BILINEAR, expand=False)
            
            # Random horizontal flip
            if random.random() < 0.3:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Random crop and resize
            if random.random() < 0.4:
                width, height = img.size
                crop_size = min(width, height)
                crop_size = int(crop_size * random.uniform(0.8, 1.0))
                left = random.randint(0, width - crop_size)
                top = random.randint(0, height - crop_size)
                img = img.crop((left, top, left + crop_size, top + crop_size))
                img = img.resize((width, height), Image.BILINEAR)
                
            return img
        except Exception as e:
            print(f"Error in normal augmentation: {str(e)}")
            return image
    
    def _generate_anomaly_embeddings(self, samples_dict: Dict[str, List[str]], n_clusters: int = 20) -> torch.Tensor:
        """
        Generate diverse anomaly embeddings using enhanced augmentation
        """
        print("Starting anomaly embeddings generation...")
        anomaly_embeddings = []
        augmenter = AnomalyAugmenter(severity=0.5)
        
        batch_images = []
        batch_size = 16
        
        # 샘플 수 증가
        samples_per_class = 5
        augmentations_per_sample = 5  # 증가
        
        total_classes = len(samples_dict)
        processed = 0
        
        for class_name, image_paths in tqdm(samples_dict.items(), desc="Generating anomaly embeddings"):
            print(f"Processing class {class_name} ({processed+1}/{total_classes})")
            
            for img_path in image_paths[:samples_per_class]:
                try:
                    image = Image.open(img_path).convert('RGB')
                    if min(image.size) < 64:
                        continue
                    
                    # 다양한 augmentation 적용
                    for _ in range(augmentations_per_sample):
                        anomaly_image = augmenter.generate_anomaly(image.copy())
                        
                        if anomaly_image is not None:
                            batch_images.append(
                                self.model.preprocess(anomaly_image)
                            )
                            
                            if len(batch_images) >= batch_size:
                                batch = torch.stack(batch_images).to(self.model.device)
                                features = self.model.extract_features(batch)
                                anomaly_embeddings.append(features)
                                batch_images = []
                                
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
                    continue
                    
            processed += 1
        
        # 남은 배치 처리
        if batch_images:
            batch = torch.stack(batch_images).to(self.model.device)
            features = self.model.extract_features(batch)
            anomaly_embeddings.append(features)
        
        if not anomaly_embeddings:
            print("Warning: No anomaly embeddings generated")
            return torch.randn(1, 768, device=self.model.device)
        
        # 모든 임베딩 결합
        all_embeddings = torch.cat(anomaly_embeddings, dim=0)
        
        # k-means 클러스터링 적용
        try:
            print(f"Applying k-means clustering to get {n_clusters} representative embeddings...")
            embeddings_np = all_embeddings.cpu().numpy()
            
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=100,
                random_state=42
            )
            kmeans.fit(embeddings_np)
            
            cluster_centers = torch.tensor(
                kmeans.cluster_centers_,
                device=self.model.device,
                dtype=all_embeddings.dtype
            )
            
            cluster_centers = F.normalize(cluster_centers, dim=1)
            
            print(f"Generated {len(cluster_centers)} representative anomaly embeddings")
            return cluster_centers
            
        except Exception as e:
            print(f"Error in clustering: {str(e)}, using original embeddings")
            indices = torch.randperm(len(all_embeddings))[:n_clusters]
            return all_embeddings[indices]

    def _compute_final_score(self, normal_similarity: float, anomaly_similarity: float, memory_score: float) -> float:
        # 정상 샘플의 confidence를 더 강하게 반영
        normal_conf = torch.sigmoid(torch.tensor((normal_similarity - 0.7) * 12.0)).item()
        
        # anomaly confidence의 기준점 낮춤
        anomaly_conf = torch.sigmoid(torch.tensor((anomaly_similarity - 0.5) * 8.0)).item()
        
        # 가중치 재조정 (density_score 제거)
        weights = {
            'normal_conf': 1.5,    # 증가
            'anomaly_conf': 0.6,   # 감소
            'memory_score': 0.8,   # 감소
        }
        
        weighted_sum = (
            -weights['normal_conf'] * normal_conf +
            weights['anomaly_conf'] * anomaly_conf +
            weights['memory_score'] * memory_score
        )
        
        score = torch.sigmoid(torch.tensor(weighted_sum)).item()
        return float(np.clip(score, 0.1, 0.7))

    def _compute_memory_outlier_score(self, image_features: torch.Tensor) -> float:
        """Memory Bank 기반 Outlier Score 계산"""
        if self.memory_bank is None or self.memory_bank.features is None:
            return 0.0
            
        device = self.model.device
        memory_features = self.memory_bank.features.to(device)
        
        # KNN 거리 계산
        distances = torch.cdist(image_features, memory_features)
        k = min(5, len(memory_features))
        knn_distances = torch.topk(distances, k, largest=False)[0]
        
        # 거리 기반 outlier score
        outlier_score = torch.mean(knn_distances).item()
        return torch.sigmoid(torch.tensor(outlier_score * 5.0)).item()
    
    def _compute_normal_similarity(self, image_features: torch.Tensor) -> float:
        """정상 클래스와의 유사도 계산"""
        device = self.model.device
        class_similarities = []
        
        # 각 클래스 프로토타입과의 유사도 계산
        for class_name, class_embedding in self.class_embeddings.items():
            class_embedding = F.normalize(class_embedding.to(device), dim=-1)
            similarity = F.cosine_similarity(image_features, class_embedding.unsqueeze(0))
            class_similarities.append(similarity.item())
        
        # Top-2 normal similarities with weighted average
        top_2_similarities = sorted(class_similarities, reverse=True)[:2]
        return (top_2_similarities[0] * 1.5 + top_2_similarities[1]) / 2.5
    
    def _compute_anomaly_similarity(self, image_features: torch.Tensor) -> float:
        """이상 임베딩과의 유사도 계산"""
        device = self.model.device
        anomaly_embeddings = F.normalize(self.anomaly_embeddings.to(device), dim=-1)
        anomaly_similarities = F.cosine_similarity(image_features, anomaly_embeddings)
        
        # Top-k anomaly similarities
        k = min(5, len(anomaly_similarities))
        top_k_anomaly = torch.topk(anomaly_similarities, k)[0]
        return torch.mean(top_k_anomaly).item()

    def _compute_anomaly_score(self, image_features: torch.Tensor) -> Tuple[float, float, float]:
        try:
            device = self.model.device
            image_features = F.normalize(image_features, dim=-1)
            
            # 1. 기본 유사도 계산
            normal_similarity = self._compute_normal_similarity(image_features)
            anomaly_similarity = self._compute_anomaly_similarity(image_features)
            
            # 2. Memory Bank의 Outlier Score 계산
            memory_score = self._compute_memory_outlier_score(image_features)
            
            # 3. 종합 스코어 계산
            final_score = self._compute_final_score(
                normal_similarity,
                anomaly_similarity,
                memory_score,
            )
            
            return final_score, normal_similarity, anomaly_similarity
            
        except Exception as e:
            print(f"Error in compute_anomaly_score: {str(e)}")
            return None, None, None
    
    def _generate_validation_scores_per_class(self, normal_samples: Dict[str, List[str]]) -> Dict[str, Dict[str, List[float]]]:
        scores_per_class = {}
        augmenter = AnomalyAugmenter(severity=0.4)
        
        # 더 많은 샘플 사용
        max_samples_per_class = 5  # 3에서 5로 증가
        n_normal_augs = 3    # 2에서 3으로 증가
        n_anomaly_augs = 4   # 3에서 4로 증가
        
        for class_name, image_paths in normal_samples.items():
            scores_per_class[class_name] = {'normal': [], 'anomaly': []}
            selected_paths = image_paths[:max_samples_per_class]
            
            # 각 이미지에 대해
            for sample_path in selected_paths:
                try:
                    image = Image.open(sample_path).convert('RGB')
                    
                    # 1. 원본 이미지 점수
                    image_input = self.model.preprocess(image).unsqueeze(0).to(self.model.device)
                    features = self.model.extract_features(image_input)
                    score, _, _ = self._compute_anomaly_score(features)
                    if score is not None:
                        scores_per_class[class_name]['normal'].append(score)
                    
                    # 2. 정상 augmentation (약한 변형)
                    for _ in range(n_normal_augs):
                        normal_aug = self._apply_normal_augmentation(image)
                        aug_input = self.model.preprocess(normal_aug).unsqueeze(0).to(self.model.device)
                        aug_features = self.model.extract_features(aug_input)
                        aug_score, _, _ = self._compute_anomaly_score(aug_features)
                        if aug_score is not None:
                            scores_per_class[class_name]['normal'].append(aug_score)
                    
                    # 3. 비정상 augmentation (강한 변형)
                    for _ in range(n_anomaly_augs):
                        anomaly_image = augmenter.generate_anomaly(image)
                        if anomaly_image is not None:
                            anomaly_input = self.model.preprocess(anomaly_image).unsqueeze(0).to(self.model.device)
                            anomaly_features = self.model.extract_features(anomaly_input)
                            anomaly_score, _, _ = self._compute_anomaly_score(anomaly_features)
                            if anomaly_score is not None:
                                scores_per_class[class_name]['anomaly'].append(anomaly_score)
                                
                except Exception as e:
                    print(f"Error processing validation sample {sample_path}: {str(e)}")
                    continue
        
        return scores_per_class
    
    def _find_optimal_thresholds_per_class(self, scores_per_class: Dict[str, Dict[str, List[float]]]) -> Dict[str, float]:
        thresholds = {}
        
        for class_name, scores in scores_per_class.items():
            if not scores['normal'] or not scores['anomaly']:
                thresholds[class_name] = 0.5
                continue
            
            normal_scores = np.array(scores['normal'])
            anomaly_scores = np.array(scores['anomaly'])
            
            # 1. 각 분포의 통계치 계산
            normal_mean = np.mean(normal_scores)
            normal_std = np.std(normal_scores)
            anomaly_mean = np.mean(anomaly_scores)
            anomaly_std = np.std(anomaly_scores)
            
            # 2. 두 분포 간의 거리 계산
            distribution_distance = abs(normal_mean - anomaly_mean)
            
            # 3. 두 분포의 겹치는 정도를 고려
            overlap_factor = min(normal_std, anomaly_std) / distribution_distance
            
            # 4. 기본 threshold 계산 (두 분포 사이)
            base_threshold = (normal_mean * anomaly_std + anomaly_mean * normal_std) / (normal_std + anomaly_std)
            
            # 5. overlap에 따른 조정
            if overlap_factor < 0.3:  # 분포가 잘 분리됨
                threshold = base_threshold
            else:  # 분포가 많이 겹침
                # normal 쪽으로 더 보수적으로 조정
                threshold = normal_mean + normal_std * (1 - overlap_factor)
            
            # 6. 극단값 필터링을 위한 percentile 사용
            normal_percentile_95 = np.percentile(normal_scores, 95)
            anomaly_percentile_5 = np.percentile(anomaly_scores, 5)
            
            # 7. 최종 threshold 결정
            if threshold > normal_percentile_95:
                threshold = normal_percentile_95
            elif threshold < anomaly_percentile_5:
                threshold = anomaly_percentile_5
                
            thresholds[class_name] = float(threshold)
            
        return thresholds
    
    # def _find_optimal_threshold(self, scores: Dict[str, List[float]]) -> float:
    #     """데이터 기반 최적 임계값 탐색"""
    #     if not scores['normal'] or not scores['anomaly']:
    #         return 0.5
        
    #     normal_scores = np.array(scores['normal'])
    #     anomaly_scores = np.array(scores['anomaly'])
        
    #     # 분포 통계 계산
    #     normal_mean = np.mean(normal_scores)
    #     normal_std = np.std(normal_scores)
    #     anomaly_mean = np.mean(anomaly_scores)
    #     anomaly_std = np.std(anomaly_scores)
        
    #     # GMM을 사용한 임계값 최적화
    #     all_scores = np.concatenate([normal_scores, anomaly_scores])
    #     gmm = GaussianMixture(n_components=2, random_state=42)
    #     gmm.fit(all_scores.reshape(-1, 1))
        
    #     # 두 가우시안의 교차점을 임계값으로 사용
    #     threshold = (normal_mean + anomaly_mean) / 2
        
    #     # 추가 제약 조건
    #     if normal_mean < anomaly_mean:
    #         # anomaly scores가 더 높은 경우
    #         threshold = normal_mean + normal_std * 1.5
    #     else:
    #         # normal scores가 더 높은 경우
    #         threshold = normal_mean - normal_std * 1.5
        
    #     # Threshold clipping
    #     threshold = np.clip(threshold, 0.35, 0.65)
        
    #     return float(threshold)