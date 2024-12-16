import torch
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Dict, List, Tuple
from PIL import Image
from tqdm import tqdm
from utils.augmentation.anomaly_augmenter import AnomalyAugmenter

class AnomalyDetector:
    def __init__(self, model, threshold: float = 0.2):
        """
        Initialize anomaly detector.
        
        Args:
            model: CLIP model instance
            threshold: Threshold for anomaly detection (default: 0.2)
        """
        self.model = model
        self.threshold = threshold
        self.class_embeddings = None
        self.anomaly_embeddings = None
        self.memory_bank = {}
        
    def prepare(self, normal_samples: Dict[str, List[str]]) -> None:
        """
        Prepare the detector by computing necessary embeddings.
        
        Args:
            normal_samples: Dictionary containing paths of normal images for each class
        """
        self.class_embeddings = self._compute_class_embeddings(normal_samples)
        self.anomaly_embeddings = self._generate_anomaly_embeddings(normal_samples)
        self._initialize_memory_bank(normal_samples)

    def predict(self, image: torch.Tensor) -> Dict:
        """
        Predict whether an image is anomalous.
        
        Args:
            image: Input image tensor
            
        Returns:
            Dict: Prediction results including predicted label and scores
        """
        try:
            features = self.model.extract_features(image)
            if features is None:
                raise ValueError("Failed to extract features from image")
                
            score, normal_sim, anomaly_sim = self._compute_anomaly_score(features)
            if any(x is None for x in [score, normal_sim, anomaly_sim]):
                raise ValueError("Failed to compute anomaly score")
                
            is_anomaly = score > self.threshold
            
            return {
                'predicted_label': 'anomaly' if is_anomaly else 'normal',
                'anomaly_score': float(score),
                'normal_similarity': float(normal_sim),
                'anomaly_similarity': float(anomaly_sim),
                'is_anomaly': is_anomaly,
                'threshold': float(self.threshold)
            }
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return {
                'predicted_label': 'error',
                'anomaly_score': 1.0,
                'normal_similarity': 0.0,
                'anomaly_similarity': 1.0,
                'is_anomaly': True,
                'threshold': float(self.threshold)
            }

    def _compute_class_embeddings(
        self, 
        samples_dict: Dict[str, List[str]]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute embeddings for each normal class.
        
        Args:
            samples_dict: Dictionary of normal sample paths
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary of class embeddings
        """
        class_embeddings = {}
        
        for class_name, image_paths in tqdm(samples_dict.items(), 
                                          desc="Computing class embeddings"):
            embeddings = []
            for img_path in image_paths:
                try:
                    image = Image.open(img_path).convert('RGB')
                    image_input = self.model.preprocess(image).unsqueeze(0).to(self.model.device)
                    features = self.model.extract_features(image_input)
                    embeddings.append(features)
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
                    continue
            
            if embeddings:
                # Robust averaging with outlier removal
                stacked_embeddings = torch.cat(embeddings, dim=0)
                mean_embedding = torch.mean(stacked_embeddings, dim=0)
                distances = torch.norm(stacked_embeddings - mean_embedding, dim=1)
                mask = distances < (distances.mean() + 2 * distances.std())
                filtered_embeddings = stacked_embeddings[mask]
                
                class_embedding = torch.mean(filtered_embeddings, dim=0)
                class_embeddings[class_name] = class_embedding / class_embedding.norm(dim=-1, keepdim=True)
        
        return class_embeddings

    def _initialize_memory_bank(self, samples_dict: Dict[str, List[str]]) -> None:
        """
        Initialize memory bank with normal samples.
        
        Args:
            samples_dict: Dictionary of normal sample paths
        """
        for class_name, image_paths in samples_dict.items():
            class_features = []
            for img_path in image_paths:
                try:
                    image = Image.open(img_path).convert('RGB')
                    image_input = self.model.preprocess(image).unsqueeze(0).to(self.model.device)
                    features = self.model.extract_features(image_input)
                    class_features.append(features)
                except Exception:
                    continue
                    
            if class_features:
                self.memory_bank[class_name] = torch.cat(class_features, dim=0)

    def _generate_anomaly_embeddings(
        self, 
        samples_dict: Dict[str, List[str]], 
        n_anomalies: int = 3
    ) -> torch.Tensor:
        """
        Generate anomaly embeddings using augmentation.
        
        Args:
            samples_dict: Dictionary of normal sample paths
            n_anomalies: Number of anomalies to generate per class
            
        Returns:
            torch.Tensor: Tensor of anomaly embeddings
        """
        anomaly_embeddings = []
        augmenter = AnomalyAugmenter(severity=0.4)
        
        for class_name, image_paths in tqdm(samples_dict.items(), 
                                          desc="Generating anomaly embeddings"):
            for img_path in image_paths[:n_anomalies]:
                try:
                    image = Image.open(img_path).convert('RGB')
                    anomaly_image = augmenter.generate_anomaly(image)
                    image_input = self.model.preprocess(anomaly_image).unsqueeze(0).to(self.model.device)
                    features = self.model.extract_features(image_input)
                    anomaly_embeddings.append(features)
                except Exception as e:
                    print(f"Error generating anomaly for {img_path}: {str(e)}")
                    continue
        
        if not anomaly_embeddings:
            raise ValueError("Failed to generate any anomaly embeddings")
            
        return torch.cat(anomaly_embeddings, dim=0)

    def _compute_specific_anomaly_features(self, image: torch.Tensor) -> Dict[str, float]:
        """
        Compute specific anomaly features based on known characteristics.
        
        Args:
            image: Input image tensor
            
        Returns:
            Dict[str, float]: Dictionary of specific anomaly scores
        """
        img_np = image.cpu().numpy().squeeze().transpose(1, 2, 0)
        img_np = (img_np * 255).astype(np.uint8)
        
        scores = {}
        
        # Red dot detection
        scores['red_score'] = self._detect_red_regions(img_np)
        
        # Deformation detection
        scores['deformation_score'] = self._detect_deformation(img_np)
        
        # Missing parts detection
        scores['missing_parts_score'] = self._detect_missing_parts(img_np)
        
        return scores

    def _detect_red_regions(self, img: np.ndarray) -> float:
        """Detect red regions in the image"""
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Red color ranges in HSV
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.add(mask1, mask2)
        
        return np.sum(red_mask > 0) / red_mask.size

    def _detect_deformation(self, img: np.ndarray) -> float:
        """Detect deformation using edge analysis"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy)
        
        hist = np.histogram(ang[mag > 0], bins=36)[0]
        hist = hist / (hist.sum() + 1e-10)
        uniformity = -np.sum(hist * np.log2(hist + 1e-10))
        
        return uniformity / 5.0

    def _detect_missing_parts(self, img: np.ndarray) -> float:
        """Detect missing parts using local intensity analysis"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        local_std = cv2.Laplacian(gray, cv2.CV_64F).std()
        score = local_std / 128.0
        return min(score, 1.0)

    def _compute_anomaly_score(self, image_features: torch.Tensor) -> Tuple[float, float, float]:
        """
        Compute anomaly score using multiple similarity measures.
        """
        try:
            if self.class_embeddings is None or self.anomaly_embeddings is None:
                raise ValueError("Embeddings not initialized. Call prepare() first.")
            
            # Ensure correct feature dimensions
            if len(image_features.shape) == 3:
                image_features = image_features.squeeze(0)

            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Calculate normal similarities
            normal_similarities = []
            for class_embedding in self.class_embeddings.values():
                similarity = F.cosine_similarity(image_features.unsqueeze(0), 
                                            class_embedding.unsqueeze(0))
                normal_similarities.append(similarity.item())
            
            max_normal_similarity = max(normal_similarities)
            
            # Calculate memory bank similarity
            memory_similarities = []
            for memory_features in self.memory_bank.values():
                similarities = F.cosine_similarity(
                    image_features.unsqueeze(0),
                    memory_features
                )
                memory_similarities.append(similarities.max().item())
            
            memory_similarity = max(memory_similarities) if memory_similarities else max_normal_similarity
            
            # Calculate anomaly similarity
            anomaly_similarities = F.cosine_similarity(
                image_features.unsqueeze(0),
                self.anomaly_embeddings
            )
            mean_anomaly_similarity = anomaly_similarities.mean().item()
            
            # Get specific anomaly features
            specific_scores = self._compute_specific_anomaly_features(image_features.unsqueeze(0))
            specific_score = (
                0.4 * specific_scores['red_score'] +
                0.3 * specific_scores['deformation_score'] +
                0.3 * specific_scores['missing_parts_score']
            )
            
            # Compute final scores
            final_normal_similarity = max(max_normal_similarity, memory_similarity)
            final_anomaly_similarity = 0.7 * mean_anomaly_similarity + 0.3 * specific_score
            
            # Calculate final anomaly score
            anomaly_score = 1.0 - final_normal_similarity + final_anomaly_similarity
            
            return anomaly_score, final_normal_similarity, final_anomaly_similarity
            
        except Exception as e:
            print(f"Error in compute_anomaly_score: {str(e)}")
            return None, None, None