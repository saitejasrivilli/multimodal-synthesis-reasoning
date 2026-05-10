import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class MultimodalFusion(nn.Module):
    def __init__(self, embedding_dim=256, hidden_dim=512, vision_dim=256, language_dim=256):
        super().__init__()
        
        # No projection needed - both are 256
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        self.layer_norm = nn.LayerNorm(embedding_dim)
        logger.info("Initialized MultimodalFusion")
    
    def forward(self, vision_features, language_features):
        v = vision_features.unsqueeze(1)
        l = language_features.unsqueeze(1)
        
        attended_v, _ = self.attention(v, l, l)
        fused = torch.cat([attended_v, l], dim=-1)
        fused = fused.squeeze(1)
        fused_embedding = self.fusion_mlp(fused)
        fused_embedding = self.layer_norm(fused_embedding + v.squeeze(1))
        
        return fused_embedding
