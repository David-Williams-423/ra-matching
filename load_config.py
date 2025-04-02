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
    try:
        # Check if file exists
        if not os.path.exists(config_path):
            print(f"Warning: Configuration file '{config_path}' not found. Using default values.")
            return {
                'faculty_weight': 0.5,
                'penalty_factor': 0.5
            }
        
        # Open and load the YAML file
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
            
        # Validate required parameters
        required_params = ['faculty_weight', 'penalty_factor']
        for param in required_params:
            if param not in config:
                print(f"Warning: Missing parameter '{param}' in config. Using default value.")
                config[param] = 0.5
                
        # Validate parameter ranges
        if not 0 <= config['faculty_weight'] <= 1:
            print(f"Warning: faculty_weight must be between 0 and 1. Using default value.")
            config['faculty_weight'] = 0.5
            
        if not 0 <= config['penalty_factor'] <= 1:
            print(f"Warning: penalty_factor must be between 0 and 1. Using default value.")
            config['penalty_factor'] = 0.5
            
        return config
    
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        print("Using default configuration values.")
        return {
            'faculty_weight': 0.5,
            'penalty_factor': 0.5
        }