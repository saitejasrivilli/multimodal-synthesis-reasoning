import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class MultimodalFusion(nn.Module):
    """Real multimodal fusion with cross-attention"""
    
    def __init__(self, embedding_dim=256, hidden_dim=512):
        super().__init__()
        
        # Projection layers to ensure consistent dimensions
        self.vision_proj = nn.Linear(256, embedding_dim)
        self.language_proj = nn.Linear(256, embedding_dim)
        
        # Cross-attention (fix: embedding_dim matches input)
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )
        
        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        logger.info("Initialized MultimodalFusion with cross-attention")
    
    def forward(self, vision_features, language_features):
        """Fuse vision and language with attention"""
        # Project to same dimension
        v = self.vision_proj(vision_features).unsqueeze(1)  # (batch, 1, embedding_dim)
        l = self.language_proj(language_features).unsqueeze(1)  # (batch, 1, embedding_dim)
        
        # Cross-attention: vision attends to language
        attended_v, _ = self.attention(v, l, l)  # (batch, 1, embedding_dim)
        
        # Concatenate and fuse
        fused = torch.cat([attended_v, l], dim=-1)  # (batch, 1, embedding_dim*2)
        fused = fused.squeeze(1)  # (batch, embedding_dim*2)
        
        fused_embedding = self.fusion_mlp(fused)  # (batch, embedding_dim)
        
        # Residual connection with proper dimension
        fused_embedding = self.layer_norm(fused_embedding + v.squeeze(1))
        
        return fused_embedding
