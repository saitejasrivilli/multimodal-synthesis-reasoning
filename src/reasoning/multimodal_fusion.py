import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

class MultimodalFusion:
    """Fuse visual and language embeddings"""
    
    def __init__(self, hidden_dim: int = 768):
        self.hidden_dim = hidden_dim
        logger.info(f"Initialized multimodal fusion (hidden_dim={hidden_dim})")
    
    def fuse(self, visual_embedding: List[float], text_embedding: List[float]) -> List[float]:
        """Fuse visual and text embeddings"""
        # Simple concatenation for now
        return visual_embedding[:384] + text_embedding[:384]
    
    def fuse_batch(self, visual: List[List[float]], text: List[List[float]]) -> List[List[float]]:
        """Fuse batches"""
        return [self.fuse(v, t) for v, t in zip(visual, text)]
