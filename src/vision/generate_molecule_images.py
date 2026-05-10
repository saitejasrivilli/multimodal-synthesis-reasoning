import torch
import numpy as np
from PIL import Image, ImageDraw
import io
import logging

logger = logging.getLogger(__name__)

class MoleculeImageGenerator:
    """Generate realistic molecular structure images"""
    
    def __init__(self):
        self.molecules = {
            "Aspirin": {
                "structure": "Acetylsalicylic acid",
                "atoms": ["C", "C", "C", "C", "C", "C", "O", "O", "C", "H"],
                "bonds": [(0,1), (1,2), (2,3), (3,4), (4,5), (5,0), (0,6), (6,7), (6,8)],
                "color": (100, 150, 200)
            },
            "Ibuprofen": {
                "structure": "2-(4-isobutylphenyl)propionic acid",
                "atoms": ["C", "C", "C", "C", "C", "C", "C", "C", "O", "O"],
                "bonds": [(0,1), (1,2), (2,3), (3,4), (4,5), (5,0), (1,6), (6,7), (7,8), (8,9)],
                "color": (150, 100, 200)
            },
            "Paracetamol": {
                "structure": "N-(4-hydroxyphenyl)acetamide",
                "atoms": ["C", "C", "C", "C", "C", "C", "N", "O", "O", "H"],
                "bonds": [(0,1), (1,2), (2,3), (3,4), (4,5), (5,0), (2,6), (6,7), (7,8)],
                "color": (200, 150, 100)
            },
            "Naproxen": {
                "structure": "2-methoxy-N-methyl-6-methyl-1,2,3,4-tetrahydronaphthalen-2-carboxylic acid",
                "atoms": ["C"]*15 + ["O"]*3 + ["N"],
                "bonds": [(i, i+1) for i in range(15)],
                "color": (150, 200, 100)
            },
            "Ketoprofen": {
                "structure": "2-(3-benzoylphenyl)propionic acid",
                "atoms": ["C"]*15 + ["O"]*3,
                "bonds": [(i, i+1) for i in range(15)],
                "color": (200, 100, 150)
            }
        }
        logger.info("Initialized MoleculeImageGenerator")
    
    def generate_image(self, molecule_name, size=224):
        """Generate a simple but realistic molecule structure image"""
        
        mol_info = self.molecules.get(molecule_name)
        if not mol_info:
            return self._generate_blank(molecule_name, size)
        
        # Create image
        img = Image.new('RGB', (size, size), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Draw benzene ring (common in these molecules)
        center_x, center_y = size // 2, size // 2
        radius = size // 6
        
        # Hexagon points
        points = []
        for i in range(6):
            angle = i * 60 * np.pi / 180
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            points.append((x, y))
        
        # Draw bonds (black lines)
        for i in range(6):
            x1, y1 = points[i]
            x2, y2 = points[(i+1) % 6]
            draw.line([(x1, y1), (x2, y2)], fill=(0, 0, 0), width=3)
        
        # Draw atoms (circles)
        atom_color = mol_info["color"]
        for x, y in points:
            draw.ellipse(
                [(x-8, y-8), (x+8, y+8)],
                fill=atom_color,
                outline=(0, 0, 0),
                width=2
            )
        
        # Add side chains (extended arms)
        for i in range(0, 6, 2):
            x1, y1 = points[i]
            x2 = x1 + (x1 - center_x) * 0.7
            y2 = y1 + (y1 - center_y) * 0.7
            draw.line([(x1, y1), (x2, y2)], fill=(0, 0, 0), width=2)
            draw.ellipse([(x2-6, y2-6), (x2+6, y2+6)], fill=(100, 100, 100))
        
        # Add molecule name
        draw.text((10, 10), molecule_name, fill=(0, 0, 0))
        
        # Convert to tensor
        img_array = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # (3, 224, 224)
        
        return tensor
    
    def _generate_blank(self, name, size):
        """Generate blank image with molecule name"""
        img = Image.new('RGB', (size, size), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((50, size//2), name, fill=(0, 0, 0))
        img_array = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        return tensor
