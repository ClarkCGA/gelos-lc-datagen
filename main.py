import argparse
from src.data_cleaning import clean_data
from src.lc_generation import generate_dataset

def main():
    parser = argparse.ArgumentParser(description='Run GFM benchmark pipeline')
    parser.add_argument('--config', '-c', 
                       default='config.yml',
                       help='Path to config file (default: config.yml)')
    
    args = parser.parse_args()
    
    generate_dataset(args.config)
    clean_data(args.config)
    
if __name__ == '__main__':
    main()