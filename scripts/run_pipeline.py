#!/usr/bin/env python3
"""Project 2: Multimodal Synthesis Reasoning - COMPLETE"""

import sys
import logging
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vision.real_image_encoder import RealReactionImageEncoder
from src.language.real_synthesis_predictor import RealSynthesisPredictor
from src.evaluation.real_synthesis_evaluator import evaluate_synthesis_prediction

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    Path("results").mkdir(exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Project 2: Multimodal Synthesis Reasoning - COMPLETE")
    logger.info("=" * 80)
    
    # 1. Load synthesis data
    logger.info("[1/4] Loading synthesis procedures...")
    all_procs = []
    with open("data/labeled/synthesis_procedures.jsonl") as f:
        for line in f:
            all_procs.append(json.loads(line))
    
    molecules = list(set(p["molecule"] for p in all_procs))
    logger.info(f"✓ Loaded {len(all_procs)} procedures for {len(molecules)} molecules")
    
    # 2. Initialize vision encoder
    logger.info("[2/4] Initializing vision encoder...")
    vision = RealReactionImageEncoder()
    
    # 3. Initialize synthesis predictor
    logger.info("[3/4] Initializing synthesis predictor...")
    predictor = RealSynthesisPredictor()
    
    # 4. Predict and evaluate
    logger.info("[4/4] Predicting synthesis procedures...")
    predictions = predictor.predict_batch(molecules)
    ground_truth = [p for p in all_procs if p["molecule"] in molecules][:len(molecules)]
    
    eval_results = evaluate_synthesis_prediction(predictions, ground_truth)
    
    # Save results
    final_results = {
        "timestamp": datetime.now().isoformat(),
        "molecules_tested": len(molecules),
        "molecules": molecules,
        "evaluation": eval_results,
        "predictions": predictions[:5]  # Show first 5
    }
    
    with open("results/synthesis_results.json", 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logger.info("=" * 80)
    logger.info("✓ PIPELINE COMPLETE")
    logger.info(f"  Molecules: {len(molecules)}")
    logger.info(f"  Step accuracy: {eval_results['step_accuracy']:.1%}")
    logger.info(f"  Condition accuracy: {eval_results['condition_accuracy']:.1%}")
    logger.info(f"  Successful predictions: {eval_results['successful_predictions']}/{len(molecules)}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
