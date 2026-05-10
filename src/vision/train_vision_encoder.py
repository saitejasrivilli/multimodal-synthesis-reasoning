import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import logging

logger = logging.getLogger(__name__)

class VisionEncoderTrainer:
    """Train vision encoder on molecule images"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.model = models.resnet18(pretrained=True)
        # Replace final layer for chemistry features
        self.model.fc = nn.Linear(512, 256)
        self.model = self.model.to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_fn = nn.MSELoss()
        
        logger.info("Initialized VisionEncoderTrainer")
    
    def train(self, images, targets, epochs=10, batch_size=8):
        """Train vision encoder"""
        self.model.train()
        losses = []
        
        n_batches = len(images) // batch_size
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = start + batch_size
                
                batch_images = images[start:end].to(self.device)
                batch_targets = targets[start:end].to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_images)
                loss = self.loss_fn(outputs, batch_targets)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)
            
            if (epoch + 1) % 3 == 0:
                logger.info(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")
        
        return losses
