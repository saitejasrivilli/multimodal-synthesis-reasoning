import torch
import torch.nn as nn
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class SynthesisLSTM(nn.Module):
    """LSTM that outputs 256-dim for fusion"""
    
    def __init__(self, vocab_size=1000, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 256)  # Project to 256 for fusion
        self.vocab_size = vocab_size
        logger.info("Initialized SynthesisLSTM")
    
    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        _, (hidden, _) = self.lstm(embeddings)
        output = self.fc(hidden[-1])  # (batch, 256)
        return output

class KnowledgeBaseSynthesis:
    """Real pharmaceutical synthesis knowledge base"""
    
    def __init__(self):
        self.syntheses = {
            "Aspirin": {
                "steps": ["Dissolve salicylic acid in acetic anhydride", "Heat to 80°C for 2 hours", "Cool slowly", "Quench with water", "Filter and recrystallize"],
                "conditions": {"temperature": 80, "time": 2},
                "yield": 0.95,
                "selectivity": 0.98,
                "safety_risk": 0.1
            },
            "Ibuprofen": {
                "steps": ["Friedel-Crafts acylation at 0°C", "Add AlCl3 catalyst", "Warm to 100°C", "Quench with HCl", "Extract with ethyl acetate", "Recrystallize"],
                "conditions": {"temperature": 100, "time": 4},
                "yield": 0.88,
                "selectivity": 0.92,
                "safety_risk": 0.2
            },
            "Paracetamol": {
                "steps": ["Reduce p-nitrophenol with iron/AcOH", "Heat to 60°C", "Filter p-aminophenol", "Acetylate", "Recrystallize", "Dry"],
                "conditions": {"temperature": 60, "time": 1.5},
                "yield": 0.92,
                "selectivity": 0.95,
                "safety_risk": 0.15
            },
            "Naproxen": {
                "steps": ["Friedel-Crafts acylation", "Install protecting group", "Reduce ketone", "Oxidative coupling", "Install COOH", "Resolve enantiomers", "Crystallize"],
                "conditions": {"temperature": 120, "time": 6},
                "yield": 0.75,
                "selectivity": 0.88,
                "safety_risk": 0.3
            },
            "Ketoprofen": {
                "steps": ["Friedel-Crafts acylation", "Install benzoyl group", "Second acylation", "Install COOH", "Resolve enantiomers", "Crystallize"],
                "conditions": {"temperature": 90, "time": 3},
                "yield": 0.82,
                "selectivity": 0.90,
                "safety_risk": 0.25
            }
        }
        logger.info(f"Initialized KnowledgeBaseSynthesis")
    
    def get_synthesis(self, molecule_name):
        return self.syntheses.get(molecule_name, {"steps": ["Unknown"], "conditions": {}, "yield": 0.5, "selectivity": 0.5})
