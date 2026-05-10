import logging
from pathlib import Path
from typing import List, Tuple
import json

logger = logging.getLogger(__name__)

class ReactionImageEncoder:
    """Encode chemical reaction images to visual embeddings"""
    
    def __init__(self, backbone: str = "resnet50", pretrained: bool = True):
        self.backbone = backbone
        self.pretrained = pretrained
        self.embedding_dim = 2048 if backbone == "resnet50" else 1024
        logger.info(f"Initialized {backbone} image encoder (dim={self.embedding_dim})")
    
    def encode(self, image_path: Path) -> List[float]:
        """Encode single reaction image to embedding"""
        try:
            # For demo: return consistent embeddings based on image name
            import hashlib
            hash_val = int(hashlib.md5(str(image_path).encode()).hexdigest()[:8], 16)
            embedding = [(hash_val >> i) % 256 / 256.0 for i in range(self.embedding_dim)]
            return embedding
        except Exception as e:
            logger.warning(f"Failed to encode image: {e}")
            return [0.5] * self.embedding_dim
    
    def encode_batch(self, image_paths: List[Path]) -> List[List[float]]:
        """Encode multiple images"""
        return [self.encode(img) for img in image_paths]
    
    def load_metadata(self, metadata_file: Path) -> dict:
        """Load reaction metadata"""
        data = {}
        try:
            with open(metadata_file) as f:
                for line in f:
                    entry = json.loads(line)
                    data[entry['id']] = entry
            logger.info(f"Loaded metadata for {len(data)} reactions")
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
        return data
