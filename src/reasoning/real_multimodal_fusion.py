import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class MultimodalFusion(nn.Module):
    """Real multimodal fusion with cross-attention"""
    
    def __init__(self, embedding_dim=256, hidden_dim=512, vision_dim=256, language_dim=1000):
        super().__init__()
        
        # Projection layers - language is 1000-dim from LSTM
        self.vision_proj = nn.Linear(vision_dim, embedding_dim)
        self.language_proj = nn.Linear(language_dim, embedding_dim)
        
        # Cross-attention
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
        # Project to embedding dimension
        v = self.vision_proj(vision_features).unsqueeze(1)  # (batch, 1, embedding_dim)
        l = self.language_proj(language_features).unsqueeze(1)  # (batch, 1, embedding_dim)
        
        # Cross-attention: vision attends to language
        attended_v, _ = self.attention(v, l, l)  # (batch, 1, embedding_dim)
        
        # Concatenate and fuse
        fused = torch.cat([attended_v, l], dim=-1)  # (batch, 1, embedding_dim*2)
        fused = fused.squeeze(1)  # (batch, embedding_dim*2)
        
        fused_embedding = self.fusion_mlp(fused)  # (batch, embedding_dim)
        
        # Residual connection
        fused_embedding = self.layer_norm(fused_embedding + v.squeeze(1))
        
        return fused_embedding
