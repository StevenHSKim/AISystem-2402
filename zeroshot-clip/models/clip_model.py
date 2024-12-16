import torch
import clip
from torch import nn
import torch.nn.functional as F

class CLIPModel:
    def __init__(self, device: str):
        self.device = device
        self.model, self.preprocess = self._load_clip_model()
        # Add learnable temperature parameter
        self.temp = nn.Parameter(torch.ones([]) * 0.07)
        
    def _load_clip_model(self):
        # Load larger CLIP model for better features
        model, preprocess = clip.load('ViT-L/14', self.device)
        return model, preprocess
    
    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # Extract features with attention maps
            features = self.model.encode_image(image)
            
            # Apply feature enhancement
            enhanced_features = self._enhance_features(features)
            
            return enhanced_features
    
    def _enhance_features(self, features: torch.Tensor) -> torch.Tensor:
        # L2 normalization
        features = features / features.norm(dim=-1, keepdim=True)
        
        # Apply attention mechanism
        attention = torch.matmul(features, features.transpose(-2, -1))
        attention = F.softmax(attention / self.temp, dim=-1)
        
        # Enhance features with attention
        enhanced = torch.matmul(attention, features)
        
        # Skip connection
        enhanced = features + enhanced
        
        # Final normalization
        enhanced = enhanced / enhanced.norm(dim=-1, keepdim=True)
        
        return enhanced
    
    def compute_similarity(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        # Compute cosine similarity with learned temperature
        similarity = F.cosine_similarity(features1, features2) / self.temp
        return similarity