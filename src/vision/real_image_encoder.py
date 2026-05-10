import torch
import torch.nn as nn
from torchvision import models
import logging
from src.vision.generate_molecule_images import MoleculeImageGenerator

logger = logging.getLogger(__name__)

class RealImageEncoder(nn.Module):
    """Real vision encoder using ResNet18 on molecule images"""
    
    def __init__(self, embedding_dim=256):
        super().__init__()
        # Real ResNet18 pretrained
        self.backbone = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        self.projection = nn.Linear(512, embedding_dim)
        self.embedding_dim = embedding_dim
        self.img_generator = MoleculeImageGenerator()
        
        logger.info("Initialized RealImageEncoder with ResNet18")
    
    def forward(self, x):
        """Extract features from images"""
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        embeddings = self.projection(features)
        return embeddings
    
    def encode_molecule(self, molecule_name):
        """Generate image and encode molecule"""
        # Generate realistic molecule image
        img_tensor = self.img_generator.generate_image(molecule_name, size=224)
        img_batch = img_tensor.unsqueeze(0)
        
        # Encode with ResNet
        with torch.no_grad():
            embeddings = self.forward(img_batch)
        
        return embeddings
