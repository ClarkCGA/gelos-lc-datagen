import argparse
from src.gelos_config import GELOSConfig
from src.downloader import Downloader
from src.cleaner import Cleaner

def main():
    parser = argparse.ArgumentParser(description='Run GFM benchmark pipeline')
    parser.add_argument('--config', '-c', 
                       default='config.yml',
                       help='Path to config file (default: config.yml)')
    
    args = parser.parse_args()
    gelosconfig = GELOSConfig.from_yaml(args.config)
    
    downloader = Downloader(gelosconfig)
    downloader.download()
    
    cleaner = Cleaner(gelosconfig)
    cleaner.clean()
    
if __name__ == '__main__':
    main()