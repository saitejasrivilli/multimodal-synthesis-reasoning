import logging
import json
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

def evaluate_synthesis_prediction(predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
    """Evaluate synthesis predictions against ground truth"""
    
    results = {
        "total_molecules": len(predictions),
        "step_accuracy": 0.0,
        "condition_accuracy": 0.0,
        "yield_prediction_error": 0.0,
        "successful_predictions": 0
    }
    
    step_accs = []
    cond_accs = []
    yield_errors = []
    
    for pred, truth in zip(predictions, ground_truth):
        # Step accuracy (simple: check if steps are similar length)
        pred_steps = len(pred.get("steps", []))
        truth_steps = len(truth.get("steps", []))
        step_acc = 1.0 - abs(pred_steps - truth_steps) / max(truth_steps, 1)
        step_accs.append(max(0, step_acc))
        
        # Condition accuracy
        pred_cond = pred.get("conditions", {})
        truth_cond = truth.get("conditions", {})
        
        temp_error = abs(pred_cond.get("temperature", 70) - truth_cond.get("temperature", 70)) / 150.0
        cond_acc = 1.0 - temp_error
        cond_accs.append(max(0, cond_acc))
        
        # Yield prediction error
        pred_yield = pred.get("yield", 0.75)
        truth_yield = truth.get("yield", 0.75)
        yield_error = abs(pred_yield - truth_yield)
        yield_errors.append(yield_error)
        
        # Overall success
        if step_acc > 0.7 and cond_acc > 0.7:
            results["successful_predictions"] += 1
    
    results["step_accuracy"] = sum(step_accs) / len(step_accs) if step_accs else 0.0
    results["condition_accuracy"] = sum(cond_accs) / len(cond_accs) if cond_accs else 0.0
    results["yield_prediction_error"] = sum(yield_errors) / len(yield_errors) if yield_errors else 0.0
    
    logger.info(f"Step accuracy: {results['step_accuracy']:.1%}")
    logger.info(f"Condition accuracy: {results['condition_accuracy']:.1%}")
    logger.info(f"Yield error: {results['yield_prediction_error']:.3f}")
    
    return results
