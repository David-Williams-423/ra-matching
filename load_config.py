import yaml
import os

def load_config(config_path='config.yaml'):
    """
    Load algorithm configuration parameters from YAML file.
    
    Parameters:
    config_path (str): Path to the YAML configuration file
    
    Returns:
    dict: Configuration parameters
    """
    defaults = {
                'faculty_weight': 0.5,
                'no_rank_penalty': 0.5,
                'low_rank_penalty': 0.15
            }
    try:
        # Check if file exists
        if not os.path.exists(config_path):
            print(f"Warning: Configuration file '{config_path}' not found. Using default values.")
            return defaults
        
        # Open and load the YAML file
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
            
        # Validate required parameters
        required_params = ['faculty_weight', 'no_rank_penalty', 'low_rank_penalty']
        for param in required_params:
            if param not in config:
                print(f"Warning: Missing parameter '{param}' in config. Using default value.")
                config[param] = 0.5
                
        # Validate parameter ranges
        if not 0 <= config['faculty_weight'] <= 1:
            print(f"Warning: faculty_weight must be between 0 and 1. Using default value.")
            config['faculty_weight'] = 0.5
            
        if not 0 <= config['no_rank_penalty'] <= 1:
            print(f"Warning: no_rank_penalty must be between 0 and 1. Using default value.")
            config['no_rank_penalty'] = 0.5

        if not 0 <= config['low_rank_penalty'] <= 0.2:
            print(f"Warning: low_rank_penalty must be between 0 and 0.2. Using default value.")
            config['low_rank_penalty'] = 0.5
            
        return config
    
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        print("Using default configuration values.")
        return defaults