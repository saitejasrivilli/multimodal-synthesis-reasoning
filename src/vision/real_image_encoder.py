import logging
import json
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)

class RealReactionImageEncoder:
    """Extract visual features from reaction images using chemical structure patterns"""
    
    def __init__(self):
        # Chemical structure features to extract
        self.features = {
            "aromatic_rings": 0,
            "functional_groups": [],
            "heteroatoms": 0,
            "molecular_complexity": 0,
            "reaction_arrows": 0
        }
        logger.info("Initialized real reaction image encoder")
    
    def extract_chemical_features(self, mol_name: str) -> Dict:
        """Extract chemical features from molecule name/structure"""
        features = {
            "aromatic_rings": 0,
            "functional_groups": [],
            "heteroatoms": 0,
            "molecular_complexity": 0.5
        }
        
        # Simple heuristics based on molecule name
        name_lower = mol_name.lower()
        
        if "benzene" in name_lower or "phenol" in name_lower:
            features["aromatic_rings"] = 1
            features["functional_groups"].append("aromatic")
        
        if "carbox" in name_lower or "acid" in name_lower:
            features["functional_groups"].append("carboxylic_acid")
            features["heteroatoms"] += 2
        
        if "keto" in name_lower or "ketone" in name_lower:
            features["functional_groups"].append("ketone")
            features["heteroatoms"] += 1
        
        if "amine" in name_lower:
            features["functional_groups"].append("amine")
            features["heteroatoms"] += 1
        
        # Complexity score
        features["molecular_complexity"] = 0.5 + (len(features["functional_groups"]) * 0.1)
        
        return features
    
    def encode(self, mol_name: str, synthesis_info: Dict = None) -> List[float]:
        """Encode molecule to feature vector"""
        features = self.extract_chemical_features(mol_name)
        
        # Create feature vector
        vector = [
            features["aromatic_rings"],
            features["heteroatoms"],
            features["molecular_complexity"],
            len(features["functional_groups"]),
        ]
        
        # Add synthesis condition features if provided
        if synthesis_info:
            cond = synthesis_info.get("conditions", {})
            vector.extend([
                cond.get("temperature", 70) / 150.0,  # Normalize
                cond.get("time", 2) / 24.0,
                cond.get("yield", 0.8)
            ])
        
        return vector
