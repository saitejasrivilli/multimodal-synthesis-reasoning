from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def get_raw_data_path():
    return PROJECT_ROOT / "data" / "raw"

def get_processed_data_path():
    return PROJECT_ROOT / "data" / "processed"

def get_labeled_data_path():
    return PROJECT_ROOT / "data" / "labeled"

def get_models_checkpoint_dir(name="default"):
    path = PROJECT_ROOT / "models" / "checkpoints" / name
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_results_dir():
    path = PROJECT_ROOT / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_logs_dir():
    path = PROJECT_ROOT / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path

def initialize_directories():
    for d in [get_raw_data_path(), get_processed_data_path(), get_labeled_data_path(), 
              get_results_dir(), get_logs_dir()]:
        d.mkdir(parents=True, exist_ok=True)
