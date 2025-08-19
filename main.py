import argparse
from src.gelos_config import GELOSConfig
from src.downloader import Downloader
import shutil
from pathlib import Path
# from src.cleaner import Cleaner

def main():
    parser = argparse.ArgumentParser(description='Run GFM benchmark pipeline')
    parser.add_argument('--config', '-c', 
                       default='config.yml',
                       help='Path to config file (default: config.yml)')
    
    args = parser.parse_args()
    gelosconfig = GELOSConfig.from_yaml(args.config)
    working_directory = Path(gelosconfig.directory.working) / gelosconfig.dataset.version
    # create working directory with version number if none exists
    working_directory.mkdir(exist_ok=True)
    # copy yaml to working directory
    shutil.copy(args.config, working_directory / "config.yaml")
   
    downloader = Downloader(gelosconfig)
    downloader.download()
    
    # cleaner = Cleaner(gelosconfig)
    # cleaner.clean()
    
if __name__ == '__main__':
    main()