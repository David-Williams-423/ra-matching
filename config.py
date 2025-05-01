import yaml
import os

CONFIG_PATH = 'config.yaml'


def load_config():
    """
    Load algorithm configuration parameters from YAML file.
    
    Parameters:
    config_path (str): Path to the YAML configuration file
    
    Returns:
    dict: Configuration parameters
    """
    defaults = {
                'faculty_weight': 0.5,
                'student_no_rank_penalty': 0.5,
                'faculty_no_rank_penalty': 0.5,
                'low_rank_penalty': 0.15,
                'similarity_weight': 0.2
            }
    try:
        # Check if file exists
        if not os.path.exists(CONFIG_PATH):
            print(f"Warning: Configuration file '{CONFIG_PATH}' not found. Using default values.")
            return defaults
        
        # Open and load the YAML file
        with open(CONFIG_PATH, 'r') as config_file:
            config = yaml.safe_load(config_file)
            
        # Validate required parameters
        required_params = ['faculty_weight', 'student_no_rank_penalty', 'faculty_no_rank_penalty', 'low_rank_penalty', 'similarity_weight']
        for param in required_params:
            if param not in config:
                print(f"Warning: Missing parameter '{param}' in config. Using default value.")
                config[param] = 0.5
                
        # Validate parameter ranges
        if not 0 <= config['faculty_weight'] <= 1:
            print(f"Warning: faculty_weight must be between 0 and 1. Using default value.")
            config['faculty_weight'] = 0.5
            
        if not 0 <= config['student_no_rank_penalty'] <= 1:
            print(f"Warning: student_no_rank_penalty must be between 0 and 1. Using default value.")
            config['student_no_rank_penalty'] = 0.5

        if not 0 <= config['faculty_no_rank_penalty'] <= 1:
            print(f"Warning: faculty_no_rank_penalty must be between 0 and 1. Using default value.")
            config['faculty_no_rank_penalty'] = 0.5

        if not 0 <= config['low_rank_penalty'] <= 0.2:
            print(f"Warning: low_rank_penalty must be between 0 and 0.2. Using default value.")
            config['low_rank_penalty'] = 0.15

        if not 0 <= config['similarity_weight'] <= 0.5:
            print(f"Warning: similarity_weight must be between 0 and 0.5. Using default value.")
            config['similarity_weight'] = 0.2
            
        return config
    
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        print("Using default configuration values.")
        return defaults


def save_config(config):
    """
    Save algorithm configuration parameters to YAML file.
    
    Parameters:
    config (dict): Configuration parameters
    """
    try:
        with open(CONFIG_PATH, 'w') as config_file:
            yaml.dump(config, config_file)
    except Exception as e:
        print(f"Error saving configuration file: {e}")

def get_config_value(key):
    """
    Get a specific configuration value.
    
    Parameters:
    key (str): Configuration key
    
    Returns:
    value: Configuration value
    """
    config = load_config()
    return config.get(key, None)

def set_config_value(key, value):
    """
    Set a specific configuration value.
    
    Parameters:
    key (str): Configuration key
    value: Configuration value
    """
    config = load_config()
    config[key] = value
    save_config(config)