import argparse
from src.gelos_config import GELOSConfig
from src.downloader import Downloader
import shutil
from pathlib import Path

def setup_workspace(config_path):
    """Setup workspace and return configuration"""
    gelosconfig = GELOSConfig.from_yaml(config_path)
    working_directory = Path(gelosconfig.directory.working) / gelosconfig.dataset.version
    working_directory.mkdir(exist_ok=True)
    shutil.copy(config_path, working_directory / "config.yaml")
    return gelosconfig

def main():
    parser = argparse.ArgumentParser(description='Run GFM benchmark pipeline')
    parser.add_argument('--config', '-c', 
                       default='config.yml',
                       help='Path to config file (default: config.yml)')
    
    args = parser.parse_args()
    gelosconfig = setup_workspace(args.config)
    downloader = Downloader(gelosconfig)
    downloader.download()

if __name__ == '__main__':
    main()
