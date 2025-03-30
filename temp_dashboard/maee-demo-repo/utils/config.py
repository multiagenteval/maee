import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    @staticmethod
    def load_config(config_path: str = '.eval-config.yaml') -> Dict[str, Any]:
        """Load configuration from yaml file"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")
            
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        return config
        
    @staticmethod
    def load_experiment_config(config_path: str) -> Dict[str, Any]:
        """Load experiment configuration from yaml file"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Experiment config file not found at {config_path}")
            
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        # Validate required fields
        required_fields = ['model', 'training', 'data']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in experiment config")
                
        return config 